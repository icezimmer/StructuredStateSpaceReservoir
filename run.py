import argparse
import logging
import os
from datetime import datetime
import torch
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla import VanillaRNN, VanillaGRU
from src.models.esn.esn import ESN
from src.models.s4d.s4d import S4D
from src.models.s4r.s4r import S4R
from src.deep.stacked import StackedNetwork, StackedReservoir, StackedEchoState
from src.reservoir.readout import ReadOut
from src.ml.optimization import setup_optimizer
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier
from src.utils.saving import load_data, save_parameters, save_hyperparameters, update_results
from src.utils.check_device import check_model_device
from codecarbon import EmissionsTracker
import numpy

block_factories = {
    'S4': S4Block,
    'VanillaRNN': VanillaRNN,
    'VanillaGRU': VanillaGRU,
    'S4D': S4D,
    'ESN': ESN,
    'S4R': S4R
}

conv_classes = ['fft', 'fft-freezeD']

kernel_classes = ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC', 'V-freezeABC',
                  'miniV', 'miniV-freezeW', 'miniV-freezeA', 'miniV-freezeAW']

kernel_classes_reservoir = ['V-freezeABC', 'miniV-freezeAW']


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--device', default='cuda:1', help='Cuda device.')
    parser.add_argument('--block', choices=block_factories.keys(), default='S4D',
                        help='Block class to use for the model.')

    parser.add_argument('--layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of hidden neurons (hidden state size).')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block in ['VanillaRNN', 'VanillaGRU', 'S4', 'S4D']:
        parser.add_argument('--encoder', default='conv1d', help='Encoder model.')
        parser.add_argument('--decoder', default='conv1d', help='Decoder model.')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
        parser.add_argument('--layerdrop', type=float, default=0.0, help='Dropout the output of each layer.')

        parser.add_argument('--lr', type=float, default=0.004, help='Learning rate for NON-kernel parameters.')
        parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for NON-kernel parameters.')
        parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
        parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')
        if args.block in ['VanillaRNN', 'VanillaGRU']:
            pass
        elif args.block == 'S4':
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
        elif args.block == 'S4D':
            parser.add_argument('--conv', choices=conv_classes, default='fft', help='Skip connection matrix D.')
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernel', choices=kernel_classes, default='V',
                                help='Kernel name.')
            parser.add_argument('--mix', default='conv1d+glu', help='Inner Mixing layer.')
            parser.add_argument('--dt', type=int, default=None, help='Sampling rate (only for continuous dynamics).')
            parser.add_argument('--strong', type=float, default=0.7, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=0.95, help='Weak Stability for internal dynamics.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
    elif args.block in ['ESN', 'S4R']:
        parser.add_argument('--transient', type=int, default=-1, help='Number of fist time steps to discard.')
        if args.block == 'ESN':
            pass
        elif args.block == 'S4R':
            parser.add_argument('--encoder', default='reservoir', help='Encoder model.')
            parser.add_argument('--kernel', choices=kernel_classes_reservoir, default='V-freezeABC',
                                help='Kernel name.')
            parser.add_argument('--mix', default='reservoir+tanh', help='Inner Mixing layer.')
            parser.add_argument('--dt', type=int, default=None, help='Sampling rate (only for continuous dynamics).')
            parser.add_argument('--strong', type=float, default=0.98, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=1.0, help='Weak Stability for internal dynamics.')

    # Update args with the new conditional arguments
    args, unknown = parser.parse_known_args()

    # Conditionally add --scaleB and --scaleC if kernel starts with 'miniV'
    if hasattr(args, 'kernel') and args.kernel.startswith('V'):
        parser.add_argument('--scaleB', type=float, default=1.0, help='Scaling for the input2state matrix B.')
        parser.add_argument('--scaleC', type=float, default=1.0, help='Scaling for the state2output matrix C.')

    # Conditionally add --scaleW if kernel starts with 'miniV'
    if hasattr(args, 'kernel') and args.kernel.startswith('miniV'):
        parser.add_argument('--scaleW', type=float, default=1.0, help='Scaling for the input-output matrix W.')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == 'smnist':
        criterion = torch.nn.CrossEntropyLoss()  # classification task
        to_vec = True  # classification task (take last time as output)
        d_input = 1  # number of input features
        kernel_size = 28 * 28  # max length of input sequence
        d_output = 10  # number of classes
    elif args.task == 'pathfinder':
        criterion = torch.nn.CrossEntropyLoss()
        to_vec = True
        d_input = 1
        kernel_size = 32 * 32
        d_output = 2
    elif args.task == 'scifar10':
        criterion = torch.nn.CrossEntropyLoss()
        to_vec = True
        d_input = 3
        kernel_size = 32 * 32
        d_output = 10
    else:
        raise ValueError('Invalid task name')

    if args.block in ['VanillaRNN', 'VanillaGRU']:
        block_args = {}
    elif args.block == 'S4':
        block_args = {'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'lr': args.kernellr, 'wd': args.kernelwd}
    elif args.block == 'S4D':
        block_args = {'mixing_layer': args.mix,
                      'convolution': args.conv,
                      'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'dt': args.dt, 'strong_stability': args.strong, 'weak_stability': args.weak}

        if args.kernel in ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC',
                           'miniV', 'miniV-freezeW', 'miniV-freezeA']:
            block_args['lr'] = args.kernellr
            block_args['wd'] = args.kernelwd
        if args.kernel.startswith('V'):
            block_args['input2state_scaling'] = args.scaleB
            block_args['state2output_scaling'] = args.scaleC
        elif args.kernel.startswith('miniV'):
            block_args['input_output_scaling'] = args.scaleW
    elif args.block == 'ESN':
        block_args = {}
    elif args.block == 'S4R':
        block_args = {'mixing_layer': args.mix,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'dt': args.dt, 'strong_stability': args.strong, 'weak_stability': args.weak}
        if args.kernel.startswith('V'):
            block_args['input2state_scaling'] = args.scaleB
            block_args['state2output_scaling'] = args.scaleC
        elif args.kernel.startswith('miniV'):
            block_args['input_output_scaling'] = args.scaleW
    else:
        raise ValueError('Invalid block name')

    develop_dataloader = load_data(os.path.join('./checkpoint', 'dataloaders', args.task, 'develop_dataloader'))
    test_dataloader = load_data(os.path.join('./checkpoint', 'dataloaders', args.task, 'test_dataloader'))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.block == 'S4D':
        block_name = args.block + '_' + args.conv + '_' + args.kernel + '_' + args.mix
    else:
        block_name = args.block
    if args.block not in ['ESN', 'S4R']:
        project_name = (args.encoder + '_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.decoder)
    elif args.block == 'S4R':
        project_name = (args.encoder + '_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        'ridge')
    elif args.block == 'ESN':
        project_name = ('[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        'ridge')
    output_dir = os.path.join('./checkpoint', 'task', args.task)
    run_dir = os.path.join('./checkpoint', 'task', args.task, block_name, str(args.layers) + 'x' + str(args.neurons),
                           current_time)
    hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
    save_hyperparameters(args=args, file_path=hyperparameters_path)
    parameters_path = os.path.join(run_dir, 'parameters.txt')

    logging.basicConfig(level=logging.INFO)

    if args.block in ['VanillaRNN', 'VanillaGRU', 'S4', 'S4D']:
        logging.info('Initializing Model.')
        model = StackedNetwork(block_cls=block_factories[args.block], n_layers=args.layers,
                               d_input=d_input, d_model=args.neurons, d_output=d_output,
                               encoder=args.encoder, decoder=args.decoder,
                               to_vec=to_vec,
                               layer_dropout=args.layerdrop,
                               **block_args)

        torch.backends.cudnn.benchmark = False
        model.to(device=torch.device(args.device))

        # Initialize the tracker
        tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                   log_level="ERROR",
                                   gpu_ids=[check_model_device(model).index])

        save_parameters(model=model, file_path=parameters_path)

        train_dataloader = load_data(os.path.join('./checkpoint', 'dataloaders', args.task, 'train_dataloader'))
        val_dataloader = load_data(os.path.join('./checkpoint', 'dataloaders', args.task, 'val_dataloader'))

        optimizer = setup_optimizer(model=model, lr=args.lr, weight_decay=args.wd)
        trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                             develop_dataloader=develop_dataloader)

        logging.info('Starting Fit.')
        # Start tracking
        tracker.start()
        trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               patience=args.patience, num_epochs=args.epochs,
                               run_directory=run_dir)
        emissions = tracker.stop()
        print(f"Estimated CO2 emissions for this fit: {emissions} kg")
        # End tracking

        if args.task in ['smnist', 'pathfinder', 'scifar10']:
            eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
            eval_bc.evaluate(run_directory=run_dir, dataset_name='develop')

            eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
            eval_bc.evaluate(run_directory=run_dir, dataset_name='test')

        update_results(emissions_path=os.path.join(output_dir, 'emissions.csv'),
                       metrics_test_path=os.path.join(run_dir, 'metrics_test.json'),
                       results_path=os.path.join(output_dir, 'results.csv'))
    elif args.block in ['ESN', 'S4R']:
        logging.info('Initializing Model.')
        if args.block == 'S4R':
            model = StackedReservoir(n_layers=args.layers,
                                     d_input=d_input, d_model=args.neurons,
                                     encoder=args.encoder,
                                     transient=args.transient,
                                     **block_args)
        elif args.block == 'ESN':
            model = StackedEchoState(n_layers=args.layers,
                                     d_input=d_input, d_model=args.neurons,
                                     transient=args.transient,
                                     **block_args)

        torch.backends.cudnn.benchmark = False
        model.to(device=torch.device(args.device))

        # Initialize the tracker
        tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                   log_level="ERROR",
                                   gpu_ids=[check_model_device(model).index])

        readout = ReadOut(reservoir_model=model, develop_dataloader=develop_dataloader, d_state=args.neurons,
                          d_output=d_output, lambda_=1.0, bias=True, to_vec=to_vec)
        logging.info('Starting Fit.')
        tracker.start()
        readout.fit_()
        emissions = tracker.stop()
        print(f"Estimated CO2 emissions for this fit: {emissions} kg")
        readout.evaluate_(develop_dataloader)
        readout.evaluate_(test_dataloader)


if __name__ == '__main__':
    main()
