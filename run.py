import argparse
import logging
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from src.utils.split_data import random_split_dataset
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla import VanillaRNN, VanillaGRU
from src.models.esn.esn import ESN
from src.models.s4d.s4d import S4D
from src.models.s4r.s4r import S4R
from src.deep.stacked import StackedNetwork, StackedReservoir, StackedEchoState
from src.deep.hybrid import MLP
from src.torch_dataset.reservoir_to_nn import Reservoir2NN
from src.reservoir.readout import ReadOut
from src.ml.optimization import setup_optimizer
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier
from src.utils.saving import load_data, save_hyperparameters, update_results
from src.utils.check_device import check_model_device
from codecarbon import EmissionsTracker

block_factories = {
    'S4': S4Block,
    'RNN': VanillaRNN,
    'GRU': VanillaGRU,
    'S4D': S4D,
    'ESN': ESN,
    'S4R': S4R
}

s4_mode = ['s4d', 'diag', 's4', 'nplr', 'dplr']

conv_classes = ['fft', 'fft-freezeD']

kernel_classes = ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC', 'V-freezeABC',
                  'miniV', 'miniV-freezeW', 'miniV-freezeA', 'miniV-freezeAW']

kernel_classes_reservoir = ['Vr', 'miniVr']

readout_classes = ['offline', 'mlp', 'ssm']


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--device', default='cuda:1', help='Cuda device.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--block', choices=block_factories.keys(), default='S4R',
                        help='Block class to use for the model.')

    parser.add_argument('--layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of hidden neurons (hidden state size).')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block in ['RNN', 'GRU', 'S4', 'S4D']:
        parser.add_argument('--encoder', default='conv1d', help='Encoder model.')
        parser.add_argument('--decoder', default='conv1d', help='Decoder model.')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
        parser.add_argument('--layerdrop', type=float, default=0.0, help='Dropout the output of each layer.')
        parser.add_argument('--lr', type=float, default=0.004, help='Learning rate for NON-kernel parameters.')
        parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for NON-kernel parameters.')
        parser.add_argument('--plateau', type=float, default=0.2, help='Learning rate decay factor on Plateau.')
        parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
        parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')
        if args.block in ['RNN', 'GRU']:
            pass
        elif args.block == 'S4':
            parser.add_argument('--kernel', choices=s4_mode, default='dplr', help='Kernel name.')
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
        elif args.block == 'S4D':
            parser.add_argument('--conv', choices=conv_classes, default='fft', help='Skip connection matrix D.')
            parser.add_argument('--scaleD', type=float, default=1.0, help='Skip connection matrix D scaling.')
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernel', choices=kernel_classes, default='V', help='Kernel name.')
            parser.add_argument('--mix', default='conv1d+glu', help='Inner Mixing layer.')
            parser.add_argument('--strong', type=float, default=0.7, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=0.95, help='Weak Stability for internal dynamics.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
    elif args.block in ['ESN', 'S4R']:
        parser.add_argument('--readout', choices=readout_classes, default='mlp', help='Type of Readout.')
        if args.block == 'ESN':
            parser.add_argument('--input', type=float, default=1.0, help='Scaling of input matrix.')
            parser.add_argument('--rho', type=float, default=1.0, help='Spectral Radius of hidden state matrix.')
            parser.add_argument('--leaky', type=float, default=1.0, help='Leakage Rate for leaky integrator.')
        elif args.block == 'S4R':
            parser.add_argument('--scaleencoder', type=float, default=1.0, help='Encoder model scaling factor.')
            parser.add_argument('--scaleD', type=float, default=1.0, help='Skip connection matrix D scaling.')
            parser.add_argument('--kernel', choices=kernel_classes_reservoir, default='Vr',
                                help='Kernel name.')
            parser.add_argument('--mix', default='identity', help='Inner Mixing layer.')
            parser.add_argument('--strong', type=float, default=0.98, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=1.0, help='Weak Stability for internal dynamics.')

    # Update args with the new conditional arguments
    args, unknown = parser.parse_known_args()

    if hasattr(args, 'readout'):
        if args.readout == 'offline':
            parser.add_argument('--transient', type=int, default=-1, help='Number of first time steps to discard.')
            parser.add_argument('--ridge', type=float, default=1.0, help='Regularization for Ridge Regression.')

        if args.readout == 'mlp':
            parser.add_argument('--transient', type=int, default=-1, choices=[-1],
                                help='Number of first time steps to discard.')
            parser.add_argument('--mlplayers', type=int, default=2, help='Number of MLP layers.')
            parser.add_argument('--lr', type=float, default=0.004, help='Learning rate for MLP parameters.')
            parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for MLP parameters.')
            parser.add_argument('--plateau', type=float, default=0.2, help='Learning rate decay factor on Plateau.')
            parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
            parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')

        if args.readout == 'ssm':
            parser.add_argument('--transient', type=int, default=-128, help='Number of first time steps to discard.')
            parser.add_argument('--ssmlayers', type=int, default=1, help='Number of layers.')
            parser.add_argument('--lr', type=float, default=0.004, help='Learning rate for NON-kernel parameters.')
            parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for NON-kernel parameters.')
            parser.add_argument('--plateau', type=float, default=0.2, help='Learning rate decay factor on Plateau.')
            parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
            parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')


    # Conditionally add --dt, --scaleB and --scaleC if kernel starts with 'V'
    if hasattr(args, 'kernel'):
        if args.kernel.startswith('V'):
            parser.add_argument('--dt', type=float, default=None, help='Sampling rate (only for continuous dynamics).')
            parser.add_argument('--scaleB', type=float, default=1.0, help='Scaling for the input2state matrix B.')
            parser.add_argument('--scaleC', type=float, default=1.0, help='Scaling for the state2output matrix C.')

        # Conditionally add --scaleW if kernel starts with 'miniV'
        if hasattr(args, 'kernel') and args.kernel.startswith('miniV'):
            parser.add_argument('--scaleW', type=float, default=1.0, help='Scaling for the input-output matrix W.')

    return parser.parse_args()


# TODO: Add args for ssm readout
def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    classification_task = ['smnist', 'pmnist', 'pathfinder', 'scifar10', 'scifar10gs']
    if args.task in ['smnist', 'pmnist']:
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
    elif args.task == 'scifar10gs':
        criterion = torch.nn.CrossEntropyLoss()
        to_vec = True
        d_input = 1
        kernel_size = 32 * 32
        d_output = 10
    else:
        raise ValueError('Invalid task name')

    if args.block in ['RNN', 'GRU']:
        block_args = {}
    elif args.block == 'S4':
        block_args = {'mode':args.kernel, 'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'lr': args.kernellr, 'wd': args.kernelwd}
    elif args.block == 'S4D':
        block_args = {'mixing_layer': args.mix,
                      'convolution': args.conv,
                      'scaleD': args.scaleD,
                      'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'strong_stability': args.strong, 'weak_stability': args.weak}

        if args.kernel in ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC',
                           'miniV', 'miniV-freezeW', 'miniV-freezeA']:
            block_args['lr'] = args.kernellr
            block_args['wd'] = args.kernelwd
        if args.kernel.startswith('V'):
            block_args['dt'] = args.dt
            block_args['scaleB'] = args.scaleB
            block_args['scaleC'] = args.scaleC
        elif args.kernel.startswith('miniV'):
            block_args['scaleW'] = args.scaleW
    elif args.block == 'ESN':
        block_args = {'input_scaling': args.input, 'spectral_radius': args.rho, 'leakage_rate': args.leaky}
    elif args.block == 'S4R':
        block_args = {'mixing_layer': args.mix,
                      'scaleD': args.scaleD,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'strong_stability': args.strong, 'weak_stability': args.weak}
        if args.kernel.startswith('V'):
            block_args['dt'] = args.dt
            block_args['scaleB'] = args.scaleB
            block_args['scaleC'] = args.scaleC
        elif args.kernel.startswith('miniV'):
            block_args['scaleW'] = args.scaleW
    else:
        raise ValueError('Invalid block name')

    logging.info('Loading develop and test datasets.')
    develop_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))
    test_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'test_dataset'))
    develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.block == 'S4':
        block_name = args.block + '_' + args.kernel
    elif args.block == 'S4D':
        block_name = args.block + '_' + args.conv + '_' + args.kernel + '_' + args.mix
    elif args.block == 'S4R':
        block_name = args.block + '_' + args.kernel + '_' + args.mix
    else:
        block_name = args.block

    if args.block not in ['ESN', 'S4R']:
        project_name = (args.encoder + '_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.decoder)
    elif args.block == 'S4R':
        project_name = ('[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.readout)
    elif args.block == 'ESN':
        project_name = ('[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.readout)
    else:
        raise ValueError('Invalid block name')

    output_dir = os.path.join('./checkpoint', 'results', args.task)
    run_dir = os.path.join('./checkpoint', 'results', args.task, block_name, str(args.layers) + 'x' + str(args.neurons),
                           current_time)

    logging.info('Saving model hyper-parameters.')
    hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
    save_hyperparameters(args=args, file_path=hyperparameters_path)

    if args.block in ['RNN', 'GRU', 'S4', 'S4D']:
        logging.info('Initializing model.')
        model = StackedNetwork(block_cls=block_factories[args.block], n_layers=args.layers,
                               d_input=d_input, d_model=args.neurons, d_output=d_output,
                               encoder=args.encoder, decoder=args.decoder,
                               to_vec=to_vec,
                               layer_dropout=args.layerdrop,
                               **block_args)

        logging.info(f'Moving model to {args.device}.')
        torch.backends.cudnn.benchmark = False
        model.to(device=torch.device(args.device))

        logging.info('Splitting develop data into training and validation data.')
        train_dataset, val_dataset = random_split_dataset(develop_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

        logging.info('Setting optimizer and trainer.')
        optimizer = setup_optimizer(model=model, lr=args.lr, weight_decay=args.wd)
        trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                             develop_dataloader=develop_dataloader)

        logging.info('Tracking energy consumption.')
        tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                   log_level="ERROR",
                                   gpu_ids=[check_model_device(model).index])

        logging.info('Fitting model.')
        tracker.start()
        trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               patience=args.patience, reduce_plateau=args.plateau, num_epochs=args.epochs,
                               plot_path=os.path.join(run_dir, 'loss.png'))
        emissions = tracker.stop()
        logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

        logging.info('Saving model.')
        torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))

        if args.task in classification_task:
            logging.info('Evaluating model on develop set.')
            eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
            eval_bc.evaluate(saving_path=os.path.join(run_dir, 'develop'))

            logging.info('Evaluating model on test set.')
            eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
            eval_bc.evaluate(saving_path=os.path.join(run_dir, 'test'))

    elif args.block in ['ESN', 'S4R']:
        logging.info('Initializing model.')
        if args.block == 'S4R':
            reservoir_model = StackedReservoir(n_layers=args.layers,
                                               d_input=d_input, d_model=args.neurons,
                                               transient=args.transient,
                                               encoder_scaling=args.scaleencoder,
                                               **block_args)
            logging.info(f'Moving reservoir model to {args.device}.')
            torch.backends.cudnn.benchmark = False
            reservoir_model.to(device=torch.device(args.device))

            logging.info('Saving reservoir model.')
            torch.save(reservoir_model.state_dict(), os.path.join(run_dir, 'reservoir_model.pt'))
        elif args.block == 'ESN':
            reservoir_model = StackedEchoState(n_layers=args.layers,
                                               d_input=d_input, d_model=args.neurons,
                                               transient=args.transient,
                                               **block_args)
            logging.info(f'Moving reservoir model to {args.device}.')
            torch.backends.cudnn.benchmark = False
            reservoir_model.to(device=torch.device(args.device))

            logging.info('Saving reservoir model.')
            torch.save(reservoir_model.state_dict(), os.path.join(run_dir, 'reservoir_model.pt'))
        else:
            raise ValueError('Invalid block name')

        if args.readout == 'offline':
            logging.info('Tracking energy consumption.')
            tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                       log_level="ERROR",
                                       gpu_ids=[check_model_device(reservoir_model).index])
            logging.info('Fitting model.')
            tracker.start()
            readout = ReadOut(reservoir_model=reservoir_model, develop_dataloader=develop_dataloader,
                              d_output=d_output, to_vec=to_vec, bias=True, lambda_=args.ridge)
            readout.fit_()
            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            logging.info('Saving model.')
            torch.save(reservoir_model.state_dict(), os.path.join(run_dir, 'model.pt'))

            if args.task in classification_task:
                logging.info('Evaluating model on develop set.')
                readout.evaluate_(saving_path=os.path.join(run_dir, 'develop'))

                logging.info('Evaluating model on test set.')
                readout.evaluate_(dataloader=test_dataloader, saving_path=os.path.join(run_dir, 'test'))

        elif args.readout == 'mlp':
            model = MLP(n_layers=args.mlplayers, d_input=reservoir_model.d_output, d_output=d_output)

            logging.info(f'Moving model to {args.device}.')
            model.to(device=torch.device(args.device))

            logging.info('Setting optimizer.')
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

            logging.info('Tracking energy consumption.')
            tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                       log_level="ERROR",
                                       gpu_ids=[check_model_device(model).index])

            logging.info('Fitting model.')
            tracker.start()
            develop_dataset = Reservoir2NN(reservoir_model=reservoir_model, dataloader=develop_dataloader)
            develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=False)

            train_dataset, val_dataset = random_split_dataset(develop_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

            trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                                 develop_dataloader=develop_dataloader)

            trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                   patience=args.patience, num_epochs=args.epochs, reduce_plateau=args.plateau,
                                   plot_path=os.path.join(run_dir, 'loss.png'))
            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            logging.info('Saving model.')
            torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))

            if args.task in classification_task:
                logging.info('Evaluating model on develop set.')
                eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                eval_bc.evaluate(saving_path=os.path.join(run_dir, 'develop'))

                logging.info(f'Computing reservoir test set.')
                test_dataset = Reservoir2NN(reservoir_model=reservoir_model, dataloader=test_dataloader)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

                logging.info('Evaluating model on test set.')
                eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
                eval_bc.evaluate(saving_path=os.path.join(run_dir, 'test'))
        elif args.readout == 'ssm':
            model = StackedNetwork(block_cls=S4D, n_layers=args.ssmlayers,
                                   d_input=reservoir_model.d_output, d_model=reservoir_model.d_output, d_output=d_output,
                                   encoder='conv1d', decoder='conv1d',
                                   to_vec=to_vec,
                                   mixing_layer='conv1d+glu',
                                   convolution='fft',
                                   kernel='miniV',
                                   strong_stability=0.7,
                                   weak_stability=0.95,
                                   kernel_size=-args.transient if args.transient < 0 else kernel_size - args.transient)

            logging.info(f'Moving model to {args.device}.')
            model.to(device=torch.device(args.device))

            logging.info('Setting optimizer.')
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

            logging.info('Tracking energy consumption.')
            tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                       log_level="ERROR",
                                       gpu_ids=[check_model_device(model).index])

            logging.info('Fitting model.')
            tracker.start()
            develop_dataset = Reservoir2NN(reservoir_model=reservoir_model, dataloader=develop_dataloader)
            develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=False)

            train_dataset, val_dataset = random_split_dataset(develop_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

            trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                                 develop_dataloader=develop_dataloader)

            trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                   patience=args.patience, num_epochs=args.epochs, reduce_plateau=args.plateau,
                                   plot_path=os.path.join(run_dir, 'loss.png'))
            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            logging.info('Saving model.')
            torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))

            if args.task in classification_task:
                logging.info('Evaluating model on develop set.')
                eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                eval_bc.evaluate(saving_path=os.path.join(run_dir, 'develop'))

                logging.info(f'Computing reservoir test set.')
                test_dataset = Reservoir2NN(reservoir_model=reservoir_model, dataloader=test_dataloader)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

                logging.info('Evaluating model on test set.')
                eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
                eval_bc.evaluate(saving_path=os.path.join(run_dir, 'test'))
    else:
        raise ValueError('Invalid block name')

    logging.info('Updating results.')
    update_results(emissions_path=os.path.join(output_dir, 'emissions.csv'),
                   metrics_test_path=os.path.join(run_dir, 'test', 'metrics.json'),
                   results_path=os.path.join(output_dir, 'results.csv'))


if __name__ == '__main__':
    main()
