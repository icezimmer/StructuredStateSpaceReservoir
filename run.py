import argparse
import logging
import os
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla import VanillaRNN, VanillaGRU
from src.models.esn.esn import ESN
from src.models.ssrm.s4r import S4R
from src.task.classifier import Classifier
from src.utils.temp_data import load_temp_data
from src.utils.prints import print_parameters

block_factories = {
    'S4': S4Block,
    'VanillaRNN': VanillaRNN,
    'VanillaGRU': VanillaGRU,
    'ESN': ESN,
    'S4R': S4R
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--block', choices=block_factories.keys(), default='S4',
                        help='Block factory to use for the model.')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block in ['S4', 'S4R', 'ESN']:
        parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
        if args.block == 'S4R':
            parser.add_argument('--dt', type=int, default=None, help='Sampling rate (only for continuous dynamics).')
            parser.add_argument('--strong', type=float, default=0.9, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=1.0, help='Weak Stability for internal dynamics.')

    # Add the rest of the arguments
    parser.add_argument('--layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of hidden neurons (hidden state size).')
    parser.add_argument('--layerdrop', type=float, default=0.0, help='Dropout the output of each layer.')
    parser.add_argument('--prenorm', type=bool, default=False, help='Pre normalization or post normalization for each layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')

    # Now, parse args again to include any additional arguments
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == 'smnist':
        num_classes = 10
        num_features = 1
        kernel_size = 28 * 28
    elif args.task == 'pathfinder':
        num_classes = 2
        num_features = 1
        kernel_size = 32 * 32
    elif args.task == 'scifar10':
        num_classes = 10
        num_features = 3
        kernel_size = 32 * 32
    else:
        raise ValueError('Invalid task name')

    develop_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_develop_dataloader'))
    train_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_train_dataloader'))
    val_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_val_dataloader'))
    test_dataloader = load_temp_data(os.path.join('./checkpoint', args.task + '_test_dataloader'))

    checkpoint_path = os.path.join('./checkpoint', args.task + '_model' + '.pt')
    block_factory = block_factories[args.block]

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting Task.')

    if args.block in ['VanillaRNN', 'VanillaGRU']:
        classifier = Classifier(block_factory=block_factory, n_layers=args.layers,
                                d_input=num_features, d_model=args.neurons, num_classes=num_classes,
                                layer_dropout=args.layerdrop, pre_norm=args.prenorm)

    elif args.block == 'ESN':
        classifier = Classifier(block_factory=block_factory, n_layers=args.layers,
                                d_input=num_features, d_model=args.neurons, num_classes=num_classes,
                                layer_dropout=args.layerdrop, pre_norm=args.prenorm,
                                drop_kernel=args.kerneldrop, dropout=args.dropout)
    elif args.block == 'S4':
        classifier = Classifier(block_factory=block_factory, n_layers=args.layers,
                                d_input=num_features, d_model=args.neurons, num_classes=num_classes,
                                layer_dropout=args.layerdrop, pre_norm=args.prenorm,
                                drop_kernel=args.kerneldrop, dropout=args.dropout)
    elif args.block == 'S4R':
        classifier = Classifier(block_factory=block_factory, n_layers=args.layers,
                                d_input=num_features, d_model=args.neurons, num_classes=num_classes,
                                kernel_size=kernel_size,
                                layer_dropout=args.layerdrop, pre_norm=args.prenorm,
                                dt=args.dt, strong_stability=args.strong, weak_stability=args.weak,
                                drop_kernel=args.kerneldrop, dropout=args.dropout)
    else:
        raise ValueError('Invalid block name')

    # print_parameters(classifier.model)

    classifier.fit_model(lr=args.lr, develop_dataloader=develop_dataloader, num_epochs=args.epochs,
                         patience=args.patience, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                         checkpoint_path=checkpoint_path)

    # print_parameters(classifier.model)

    classifier.evaluate_model(develop_dataloader)
    classifier.evaluate_model(test_dataloader)


if __name__ == '__main__':
    main()
