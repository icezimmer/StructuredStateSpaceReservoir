import argparse
import logging
import os
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla_rnn import VanillaRNN
from src.models.s4d.s4d import S4D
from src.models.ssrm.s5r import S5R
from src.models.ssrm.s5fr import S5FR
from src.models.ssrm.s4r import S4R
from src.models.ssrm.s4v import S4V
from src.utils.temp_data import load_temp_data
from src.task.classifier import Classifier

block_factories = {
    'S4': S4Block,
    'VanillaRNN': VanillaRNN,
    'S4D': S4D,
    'S5R': S5R,
    'S5FR': S5FR,
    'S4R': S4R,
    'S4V': S4V
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--block', choices=block_factories.keys(), default='S4',
                        help='Block factory to use for the model.')

    args, unknown = parser.parse_known_args()

    if args.block == 'S5R' or args.block == 'S5FR' or args.block == 'S4R' or args.block == 'S4V':
        parser.add_argument('--dt', type=int, default=None, help='Sampling rate (only for continuous dynamics).')
        parser.add_argument('--strong', type=int, default=0.9, help='Strong Stability for internal dynamics.')
        parser.add_argument('--weak', type=int, default=1, help='Weak Stability for internal dynamics.')

    parser.add_argument('--layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of hidden neurons (hidden state size).')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')
    
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == 'smnist':
        num_classes = 10
        num_features_input = 1
        kernel_size = 28 * 28
    elif args.task == 'pathfinder':
        num_classes = 2
        num_features_input = 1
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

    if args.block == 'S4':
        model = Classifier(block_factory=block_factory, num_classes=num_classes, n_layers=args.layers,
                           d_model=args.neurons)
    elif args.block == 'S4D' or args.block == 'VanillaRNN':
        model = Classifier(block_factory=block_factory, num_classes=num_classes, n_layers=args.layers,
                           d_input=num_features_input, d_state=args.neurons)
    else:
        model = Classifier(block_factory=block_factory, num_classes=num_classes, n_layers=args.layers,
                           d_input=num_features_input, d_state=args.neurons,
                           kernel_size=kernel_size, strong_stability=args.strong, weak_stability=args.weak, dt=args.dt)

    # for param in model.model.parameters():
    #     print(param.data.shape)
    #     print(param)

    model.fit_model(lr=args.lr, develop_dataloader=develop_dataloader, num_epochs=args.epochs, patience=args.patience,
                    train_dataloader=train_dataloader, val_dataloader=val_dataloader, checkpoint_path=checkpoint_path)

    # for param in model.model.parameters():
    #     print(param.data.shape)
    #     print(param)

    model.evaluate_model(develop_dataloader)
    model.evaluate_model(test_dataloader)


if __name__ == '__main__':
    main()
