import argparse
import logging
import os
import torch
from src.utils.experiments import set_seed
from torch.utils.data import DataLoader
from src.utils.split_data import random_split_dataset
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla import VanillaRNN, VanillaGRU, VanillaLSTM
from src.models.esn.esn import ESN
from src.models.s4d.s4d import S4D
from src.models.rssm.rssm import RSSM
from src.deep.stacked import StackedNetwork, StackedReservoir, StackedEchoState
from src.readout.mlp import MLP
from src.torch_dataset.reservoir_to_nn import Reservoir2NN
from src.readout.ridge import Ridge
from src.ml.optimization import setup_optimizer
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier
from src.utils.saving import load_data, save_hyperparameters, update_results, update_hyperparameters
from src.utils.check_device import check_model_device
from codecarbon import EmissionsTracker

block_factories = {
    'S4': S4Block,
    'RNN': VanillaRNN,
    'GRU': VanillaGRU,
    'LSTM': VanillaLSTM,
    'S4D': S4D,
    'ESN': ESN,
    'RSSM': RSSM
}

s4_modes = ['s4d', 'diag', 's4', 'nplr', 'dplr']
s4_activations = ['tanh', 'relu', 'gelu', 'elu', 'swish', 'silu', 'glu', 'sigmoid', 'softplus']

conv_classes = ['fft', 'fft-freezeD']
kernel_classes = ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC', 'V-freezeABC',
                  'miniV', 'miniV-freezeW', 'miniV-freezeA', 'miniV-freezeAW']

kernel_classes_reservoir = ['Vr', 'miniVr']
readout_classes = ['ridge', 'mlp', 'ssm']


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save', action='store_true', help='Save results in a proper folder.')
    parser.add_argument('--tr', action='store_true', help='Development set assessment.')
    parser.add_argument('--device', default='cuda:1', help='Cuda device.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--block', choices=block_factories.keys(), default='RSSM',
                        help='Block class to use for the model.')

    parser.add_argument('--layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of hidden neurons (hidden state size).')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block in ['RNN', 'GRU', 'LSTM', 'S4', 'S4D']:
        parser.add_argument('--batch', type=int, default=128, help='Batch size')
        parser.add_argument('--encoder', default='conv1d', help='Encoder model.')
        parser.add_argument('--decoder', default='conv1d', help='Decoder model.')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
        parser.add_argument('--layerdrop', type=float, default=0.0, help='Dropout the output of each layer.')
        parser.add_argument('--lr', type=float, default=0.002, help='Learning rate for NON-kernel parameters.')
        parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for NON-kernel parameters.')
        parser.add_argument('--plateau', type=float, default=0.2, help='Learning rate decay factor on Plateau.')
        parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
        parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')
        if args.block in ['RNN', 'GRU', 'LSTM']:
            pass
        elif args.block == 'S4':
            parser.add_argument('--tiedropout', action='store_true', help='Tie dropout.')
            parser.add_argument('--bidirectional', action='store_true', help='Bidirectional.')
            parser.add_argument('--finalact', choices=s4_activations, default='glu', help='Activation.')
            parser.add_argument('--nssm', type=int, default=1, help='Kernel name.')
            parser.add_argument('--kernel', choices=s4_modes, default='dplr', help='Kernel name.')
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
        elif args.block == 'S4D':
            parser.add_argument('--conv', choices=conv_classes, default='fft', help='Skip connection matrix D.')
            parser.add_argument('--minscaleD', type=float, default=0.0, help='Skip connection matrix D min scaling.')
            parser.add_argument('--maxscaleD', type=float, default=1.0, help='Skip connection matrix D max scaling.')
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernel', choices=kernel_classes, default='V', help='Kernel name.')
            parser.add_argument('--mix', default='conv1d+glu', help='Inner Mixing layer.')
            parser.add_argument('--strong', type=float, default=0.7, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=0.95, help='Weak Stability for internal dynamics.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
    elif args.block in ['ESN', 'RSSM']:
        parser.add_argument('--rbatch', type=int, default=128, help='Batch size for Reservoir Model.')
        parser.add_argument('--readout', choices=readout_classes, default='mlp', help='Type of Readout.')
        if args.block == 'ESN':
            parser.add_argument('--inputscaling', type=float, default=1.0, help='Scaling of input matrix.')
            parser.add_argument('--biasscaling', type=float, default=0.0, help='Scaling of input matrix.')
            parser.add_argument('--rho', type=float, default=1.0, help='Spectral Radius of hidden state matrix.')
            parser.add_argument('--leaky', type=float, default=1.0, help='Leakage Rate for leaky integrator.')
        elif args.block == 'RSSM':
            parser.add_argument('--minscaleencoder', type=float, default=0.0, help='Min encoder model scaling factor.')
            parser.add_argument('--maxscaleencoder', type=float, default=1.0, help='Max encoder model scaling factor.')
            parser.add_argument('--minscaleD', type=float, default=0.0, help='Skip connection matrix D min scaling.')
            parser.add_argument('--maxscaleD', type=float, default=1.0, help='Skip connection matrix D max scaling.')
            parser.add_argument('--kernel', choices=kernel_classes_reservoir, default='Vr',
                                help='Kernel name.')
            parser.add_argument('--realfun', default='glu', help='Real function of complex variable.')
            parser.add_argument('--strong', type=float, default=0.98, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=1.0, help='Weak Stability for internal dynamics.')

    # Update args with the new conditional arguments
    args, unknown = parser.parse_known_args()

    if hasattr(args, 'readout'):
        if args.readout == 'ridge':
            parser.add_argument('--transient', type=int, default=-1, help='Number of first time steps to discard.')
            parser.add_argument('--regul', type=float, default=1.0, help='Regularization for Ridge Regression.')

        if args.readout == 'mlp':
            parser.add_argument('--batch', type=int, default=128, help='Batch size')
            parser.add_argument('--transient', type=int, default=-1, choices=[-1],
                                help='Number of first time steps to discard.')
            parser.add_argument('--mlplayers', type=int, default=2, help='Number of MLP layers.')
            parser.add_argument('--lr', type=float, default=0.004, help='Learning rate for MLP parameters.')
            parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for MLP parameters.')
            parser.add_argument('--plateau', type=float, default=0.2, help='Learning rate decay factor on Plateau.')
            parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
            parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')

        if args.readout == 'ssm':
            parser.add_argument('--batch', type=int, default=128, help='Batch size')
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
            parser.add_argument('--minscaleB', type=float, default=0.0, help='Min scaling for input2state matrix B.')
            parser.add_argument('--maxscaleB', type=float, default=1.0, help='Max scaling for input2state matrix B.')
            parser.add_argument('--minscaleC', type=float, default=0.0, help='Min scaling for state2output matrix C.')
            parser.add_argument('--maxscaleC', type=float, default=1.0, help='Max scaling for state2output matrix C.')

        # Conditionally add --scaleW if kernel starts with 'miniV'
        if hasattr(args, 'kernel') and args.kernel.startswith('miniV'):
            parser.add_argument('--scaleW', type=float, default=1.0, help='Scaling for the input-output matrix W.')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Check if cuDNN is enabled
    logging.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    logging.info(f"Setting seed: {args.seed}")
    set_seed(args.seed)

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

    if args.block in ['RNN', 'GRU', 'LSTM']:
        block_args = {}
    elif args.block == 'S4':
        block_args = {'tie_dropout': args.tiedropout, 'bidirectional': args.bidirectional,
                      'final_act': args.finalact, 'n_ssm': args.nssm,
                      'mode': args.kernel, 'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'lr': args.kernellr, 'wd': args.kernelwd}
    elif args.block == 'S4D':
        block_args = {'mixing_layer': args.mix,
                      'convolution': args.conv,
                      'min_scaleD': args.minscaleD,
                      'max_scaleD': args.maxscaleD,
                      'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'strong_stability': args.strong, 'weak_stability': args.weak}

        if args.kernel in ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC',
                           'miniV', 'miniV-freezeW', 'miniV-freezeA']:
            block_args['lr'] = args.kernellr
            block_args['wd'] = args.kernelwd
        if args.kernel.startswith('V'):
            block_args['dt'] = args.dt
            block_args['min_scaleB'] = args.minscaleB
            block_args['max_scaleB'] = args.maxscaleB
            block_args['min_scaleC'] = args.minscaleC
            block_args['max_scaleC'] = args.maxscaleC
        elif args.kernel.startswith('miniV'):
            block_args['scaleW'] = args.scaleW
    elif args.block == 'ESN':
        block_args = {'input_scaling': args.inputscaling, 'bias_scaling': args.biasscaling,
                      'spectral_radius': args.rho, 'leakage_rate': args.leaky}
    elif args.block == 'RSSM':
        block_args = {'realfun': args.realfun,
                      'min_scaleD': args.minscaleD,
                      'max_scaleD': args.maxscaleD,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'strong_stability': args.strong, 'weak_stability': args.weak}
        if args.kernel.startswith('V'):
            block_args['dt'] = args.dt
            block_args['min_scaleB'] = args.minscaleB
            block_args['max_scaleB'] = args.maxscaleB
            block_args['min_scaleC'] = args.minscaleC
            block_args['max_scaleC'] = args.maxscaleC
        elif args.kernel.startswith('miniV'):
            block_args['min_scaleW'] = args.minscaleW
            block_args['max_scaleW'] = args.maxscaleW
    else:
        raise ValueError('Invalid block name')

    if args.block == 'S4':
        block_name = args.block + '_' + args.kernel
    elif args.block == 'S4D':
        block_name = args.block + '_' + args.conv + '_' + args.kernel + '_' + args.mix
    elif args.block == 'RSSM':
        block_name = args.block + '_' + args.kernel + '_' + args.realfun
    else:
        block_name = args.block

    if args.block not in ['ESN', 'RSSM']:
        project_name = (args.encoder + '_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.decoder)
    elif args.block == 'RSSM':
        project_name = ('reservoir_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.readout)
    elif args.block == 'ESN':
        project_name = ('[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.neurons) + ']_' +
                        args.readout)
    else:
        raise ValueError('Invalid block name')

    output_dir = os.path.join('./checkpoint', 'results', args.task)
    os.makedirs(output_dir, exist_ok=True)

    if args.block in ['RNN', 'GRU', 'LSTM', 'S4', 'S4D']:
        log_file_name = args.block

        logging.info('Loading develop and test datasets.')
        develop_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))
        test_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'test_dataset'))
        develop_dataloader = DataLoader(develop_dataset,
                                        batch_size=args.batch,
                                        shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch,
                                     shuffle=False)

        logging.info('Initializing model.')
        model = StackedNetwork(block_cls=block_factories[args.block], n_layers=args.layers,
                               d_input=d_input, d_model=args.neurons, d_output=d_output,
                               encoder=args.encoder, decoder=args.decoder,
                               to_vec=to_vec,
                               layer_dropout=args.layerdrop,
                               **block_args)

        logging.info(f'Moving model to {args.device}.')
        model.to(device=torch.device(args.device))

        logging.info('Splitting develop data into training and validation data.')
        train_dataset, val_dataset = random_split_dataset(develop_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

        logging.info('Setting optimizer and trainer.')
        optimizer = setup_optimizer(model=model, lr=args.lr, weight_decay=args.wd)
        trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                             develop_dataloader=develop_dataloader)

        logging.info('Setting tracker.')
        tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                   log_level="ERROR",
                                   gpu_ids=[check_model_device(model).index])

        if args.save:
            run_dir = os.path.join(output_dir, str(tracker.run_id))
            os.makedirs(run_dir)
            plot_path = os.path.join(run_dir, 'loss.png')
            hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
            model_path = os.path.join(run_dir, 'model.pt')
            develop_path = os.path.join(run_dir, 'develop')
            test_path = os.path.join(run_dir, 'test')
        else:
            plot_path = None
            hyperparameters_path = None
            model_path = None
            develop_path = None
            test_path = None

        logging.info('[Tracking] Fitting model.')
        tracker.start()
        trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               patience=args.patience, reduce_plateau=args.plateau, num_epochs=args.epochs,
                               plot_path=plot_path)
        emissions = tracker.stop()
        logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

        if args.save:
            logging.info('Saving model hyper-parameters.')
            save_hyperparameters(args=args, file_path=hyperparameters_path)
            logging.info('Saving model.')
            torch.save(model.state_dict(), model_path)

        if args.task in classification_task:
            if args.tr:
                logging.info('Evaluating model on develop set.')
                eval_dev = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                eval_dev.evaluate(saving_path=develop_path)

            logging.info('Evaluating model on test set.')
            eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
            eval_test.evaluate(saving_path=test_path)
            scores = {'test_accuracy': eval_test.accuracy_value}
        else:
            scores = {}

    elif args.block in ['ESN', 'RSSM']:
        log_file_name = args.block + '-' + args.readout

        logging.info('Loading develop and test datasets.')
        develop_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))
        test_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'test_dataset'))
        develop_dataloader = DataLoader(develop_dataset,
                                        batch_size=args.rbatch,
                                        shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.rbatch,
                                     shuffle=False)

        logging.info('Initializing model.')
        if args.block == 'RSSM':
            reservoir_model = StackedReservoir(block_cls=block_factories[args.block],
                                               n_layers=args.layers,
                                               d_input=d_input, d_model=args.neurons,
                                               transient=args.transient,
                                               min_encoder_scaling=args.minscaleencoder,
                                               max_encoder_scaling=args.maxscaleencoder,
                                               **block_args)
            logging.info(f'Moving reservoir model to {args.device}.')
            reservoir_model.to(device=torch.device(args.device))

        elif args.block == 'ESN':
            reservoir_model = StackedEchoState(n_layers=args.layers,
                                               d_input=d_input, d_model=args.neurons,
                                               transient=args.transient,
                                               **block_args)
            logging.info(f'Moving reservoir model to {args.device}.')
            reservoir_model.to(device=torch.device(args.device))

        else:
            raise ValueError('Invalid block name')

        if args.readout == 'ridge':
            logging.info('Setting tracker.')
            tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                       log_level="ERROR",
                                       gpu_ids=[check_model_device(reservoir_model).index])

            if args.save:
                run_dir = os.path.join(output_dir, str(tracker.run_id))
                os.makedirs(run_dir)
                hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
                reservoir_model_path = os.path.join(run_dir, 'reservoir_model.pt')
                develop_path = os.path.join(run_dir, 'develop')
                test_path = os.path.join(run_dir, 'test')
            else:
                hyperparameters_path = None
                reservoir_model_path = None
                develop_path = None
                test_path = None

            logging.info('[Tracking] Fitting model.')
            tracker.start()
            readout = Ridge(reservoir_model=reservoir_model, develop_dataloader=develop_dataloader,
                            d_output=d_output, to_vec=to_vec, bias=True, lambda_=args.regul)
            readout.fit_()
            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            if args.save:
                logging.info('Saving model hyper-parameters.')
                save_hyperparameters(args=args, file_path=hyperparameters_path)
                logging.info('Saving reservoir model.')
                torch.save(reservoir_model.state_dict(), reservoir_model_path)

            if args.task in classification_task:
                if args.tr:
                    logging.info('Evaluating model on develop set.')
                    readout.evaluate_(saving_path=develop_path)

                logging.info('Evaluating model on test set.')
                readout.evaluate_(dataloader=test_dataloader, saving_path=test_path)
                scores = {'test_accuracy': readout.accuracy_value}
            else:
                scores = {}

        elif args.readout == 'mlp':
            model = MLP(n_layers=args.mlplayers, d_input=reservoir_model.d_output, d_output=d_output)

            logging.info(f'Moving model to {args.device}.')
            model.to(device=torch.device(args.device))

            logging.info('Setting optimizer.')
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

            logging.info('Setting tracker.')
            tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                       log_level="ERROR",
                                       gpu_ids=[check_model_device(model).index])

            if args.save:
                run_dir = os.path.join(output_dir, str(tracker.run_id))
                os.makedirs(run_dir)
                plot_path = os.path.join(run_dir, 'loss.png')
                hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
                reservoir_model_path = os.path.join(run_dir, 'reservoir_model.pt')
                model_path = os.path.join(run_dir, 'model.pt')
                develop_path = os.path.join(run_dir, 'develop')
                test_path = os.path.join(run_dir, 'test')
            else:
                plot_path = None
                hyperparameters_path = None
                reservoir_model_path = None
                model_path = None
                develop_path = None
                test_path = None

            logging.info('[Tracking] Fitting model.')
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
                                   plot_path=plot_path)
            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            if args.save:
                logging.info('Saving model hyper-parameters.')
                save_hyperparameters(args=args, file_path=hyperparameters_path)
                logging.info('Saving reservoir model.')
                torch.save(reservoir_model.state_dict(), reservoir_model_path)
                logging.info('Saving model.')
                torch.save(model.state_dict(), model_path)

            if args.task in classification_task:
                if args.tr:
                    logging.info('Evaluating model on develop set.')
                    eval_dev = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                    eval_dev.evaluate(saving_path=develop_path)

                logging.info(f'Computing reservoir test set.')
                test_dataset = Reservoir2NN(reservoir_model=reservoir_model, dataloader=test_dataloader)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

                logging.info('Evaluating model on test set.')
                eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
                eval_test.evaluate(saving_path=test_path)
                scores = {'test_accuracy': eval_test.accuracy_value}
            else:
                scores = {}

        elif args.readout == 'ssm':
            model = StackedNetwork(block_cls=S4D, n_layers=args.ssmlayers,
                                   d_input=reservoir_model.d_output, d_model=reservoir_model.d_output,
                                   d_output=d_output,
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

            logging.info('Setting tracker.')
            tracker = EmissionsTracker(output_dir=output_dir, project_name=project_name,
                                       log_level="ERROR",
                                       gpu_ids=[check_model_device(model).index])

            if args.save:
                run_dir = os.path.join(output_dir, str(tracker.run_id))
                os.makedirs(run_dir)
                plot_path = os.path.join(run_dir, 'loss.png')
                hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
                reservoir_model_path = os.path.join(run_dir, 'reservoir_model.pt')
                model_path = os.path.join(run_dir, 'model.pt')
                develop_path = os.path.join(run_dir, 'develop')
                test_path = os.path.join(run_dir, 'test')
            else:
                plot_path = None
                hyperparameters_path = None
                reservoir_model_path = None
                model_path = None
                develop_path = None
                test_path = None

            logging.info('[Tracking] Fitting model.')
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
                                   plot_path=plot_path)
            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            if args.save:
                logging.info('Saving model hyper-parameters.')
                save_hyperparameters(args=args, file_path=hyperparameters_path)
                logging.info('Saving reservoir model.')
                torch.save(reservoir_model.state_dict(), reservoir_model_path)
                logging.info('Saving model.')
                torch.save(model.state_dict(), model_path)

            if args.task in classification_task:
                if args.tr:
                    logging.info('Evaluating model on develop set.')
                    eval_dev = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                    eval_dev.evaluate(saving_path=develop_path)

                logging.info(f'Computing reservoir test set.')
                test_dataset = Reservoir2NN(reservoir_model=reservoir_model, dataloader=test_dataloader)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

                logging.info('Evaluating model on test set.')
                eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
                eval_test.evaluate(saving_path=test_path)
                scores = {'test_accuracy': eval_test.accuracy_value}
            else:
                scores = {}
    else:
        raise ValueError('Invalid block name')

    logging.info('Updating results.')
    update_results(emissions_path=os.path.join(output_dir, 'emissions.csv'),
                   scores=scores,
                   results_path=os.path.join(output_dir, 'results.csv'))
    update_hyperparameters(emissions_path=os.path.join(output_dir, 'emissions.csv'),
                           hyperparameters=vars(args),
                           hyperparameters_path=os.path.join(output_dir, log_file_name + '_hyperparameters.csv'))


if __name__ == '__main__':
    main()
