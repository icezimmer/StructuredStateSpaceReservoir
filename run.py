import argparse
import logging
import os
from datetime import datetime
import json
import torch
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla import VanillaRNN, VanillaGRU
from src.models.esn.esn import ESN
from src.models.ssrm.s4r import S4R
from src.kernels.vandermonde import (Vandermonde, VandermondeInputReservoir, VandermondeOutputReservoir,
                                     VandermondeInputOutputReservoir,
                                     VandermondeStateReservoir,
                                     VandermondeInputStateReservoir, VandermondeStateOutputReservoir,
                                     VandermondeFullReservoir)
from src.kernels.mini_vandermonde import (MiniVandermonde, MiniVandermondeInputOutputReservoir,
                                          MiniVandermondeStateReservoir, MiniVandermondeFullReservoir)
from torch import optim

from src.deep.residual import ResidualNetwork
from src.deep.stacked import StackedNetwork
from src.ml.optimization import setup_optimizer
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier
from src.utils.temp_data import load_temp_data
from src.utils.prints import print_buffers, print_parameters

block_factories = {
    'S4': S4Block,
    'VanillaRNN': VanillaRNN,
    'VanillaGRU': VanillaGRU,
    'ESN': ESN,
    'S4R': S4R
}

kernel_classes = {
    'V': Vandermonde,
    'V-freezeB': VandermondeInputReservoir,
    'V-freezeC': VandermondeOutputReservoir,
    'V-freezeBC': VandermondeInputOutputReservoir,
    'V-freezeA': VandermondeStateReservoir,
    'V-freezeAB': VandermondeInputStateReservoir,
    'V-freezeAC': VandermondeStateOutputReservoir,
    'V-freezeABC': VandermondeFullReservoir,
    'miniV': MiniVandermonde,
    'miniV-freezeW': MiniVandermondeInputOutputReservoir,
    'miniV-freezeA': MiniVandermondeStateReservoir,
    'miniV-freezeAW': MiniVandermondeFullReservoir,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--block', choices=block_factories.keys(), default='S4',
                        help='Block class to use for the model.')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block in ['VanillaRNN', 'VanillaGRU']:
        pass
    elif args.block == 'ESN':
        parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
    elif args.block == 'S4':
        parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
        parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
        parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
    elif args.block == 'S4R':
        parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
        parser.add_argument('--kernel', choices=kernel_classes.keys(), default='V-freezeA',
                            help='Kernel class to use for the model.')
        parser.add_argument('--dt', type=int, default=None, help='Sampling rate (only for continuous dynamics).')
        parser.add_argument('--strong', type=float, default=0.9, help='Strong Stability for internal dynamics.')
        parser.add_argument('--weak', type=float, default=1.0, help='Weak Stability for internal dynamics.')
        parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
        parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
    else:
        raise ValueError('Invalid block name')

    # Update args with the new conditional arguments
    args, unknown = parser.parse_known_args()

    # Conditionally add --scaleB and --scaleC if kernel starts with 'miniV'
    if hasattr(args, 'kernel') and args.kernel.startswith('V'):
        parser.add_argument('--scaleB', type=float, default=1.0, help='Scaling for the input2state matrix B.')
        parser.add_argument('--scaleC', type=float, default=1.0, help='Scaling for the state2output matrix C.')

    # Conditionally add --scaleW if kernel starts with 'miniV'
    if hasattr(args, 'kernel') and args.kernel.startswith('miniV'):
        parser.add_argument('--scaleW', type=float, default=1.0, help='Scaling for the input-output matrix W.')

    # Add the rest of the arguments
    parser.add_argument('--neurons', type=int, default=64, help='Number of hidden neurons (hidden state size).')
    parser.add_argument('--encoder', default='conv1d', help='Encoder model.')
    parser.add_argument('--mix', default='glu', help='Inner Mixing layer.')
    parser.add_argument('--decoder', default='conv1d', help='Decoder model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')

    parser.add_argument('--layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--layerdrop', type=float, default=0.0, help='Dropout the output of each layer.')

    parser.add_argument('--lr', type=float, default=0.004, help='Learning rate for NON-kernel parameters.')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight decay for NON-kernel parameters.')
    parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')

    # Now, parse args again to include any additional arguments
    return parser.parse_args()


def save_hyperparameters(args, save_path):
    with open(save_path, 'w') as f:
        # Convert args namespace to dictionary and save as JSON
        json.dump(vars(args), f, indent=4)


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

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join('./checkpoint', current_time)
    os.makedirs(run_dir, exist_ok=True)

    hyperparameters_path = os.path.join(run_dir, 'hyperparameters.json')
    save_hyperparameters(args, hyperparameters_path)

    develop_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_develop_dataloader'))
    train_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_train_dataloader'))
    val_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_val_dataloader'))
    test_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_test_dataloader'))

    block_cls = block_factories[args.block]

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting Task.')

    if args.block in ['VanillaRNN', 'VanillaGRU']:
        block_args = {}
    elif args.block == 'ESN':
        block_args = {'drop_kernel': args.kerneldrop, 'dropout': args.dropout}
    elif args.block == 'S4':
        block_args = {'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'lr': args.kernellr, 'wd': args.kernelwd}
    elif args.block == 'S4R':
        block_args = {'mixing_layer': args.mix,
                      'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'kernel_cls': kernel_classes[args.kernel], 'kernel_size': kernel_size,
                      'dt': args.dt, 'strong_stability': args.strong, 'weak_stability': args.weak,
                      'lr': args.kernellr, 'wd': args.kernelwd}
        if args.kernel.startswith('V'):
            block_args['input2state_scaling'] = args.scaleB
            block_args['state2output_scaling'] = args.scaleC
        elif args.kernel.startswith('miniV'):
            block_args['input_output_scaling'] = args.scaleW
        else:
            raise ValueError('Invalid kernel name')
    else:
        raise ValueError('Invalid block name')

    model = StackedNetwork(block_cls=block_cls, n_layers=args.layers,
                           d_input=d_input, d_model=args.neurons, d_output=d_output,
                           encoder=args.encoder, decoder=args.decoder,
                           to_vec=to_vec,
                           layer_dropout=args.layerdrop,
                           **block_args)

    print_parameters(model)
    print_buffers(model)

    optimizer = setup_optimizer(model=model, lr=args.lr, weight_decay=args.wd)
    trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion, develop_dataloader=develop_dataloader)
    trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                           patience=args.patience, num_epochs=args.epochs,
                           run_directory=run_dir)

    if args.task in ['smnist', 'pathfinder', 'scifar10']:
        eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
        eval_bc.evaluate(run_directory=run_dir, dataset_name='develop')

        eval_bc = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
        eval_bc.evaluate(run_directory=run_dir, dataset_name='test')


if __name__ == '__main__':
    main()
