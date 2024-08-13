import argparse
import logging
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.deep.stacked import StackedReservoir, StackedEchoState
from src.models.rssm.rssm import RSSM
from src.utils.experiments import set_seed
from src.utils.saving import load_data
from src.utils.check_device import check_model_device
from src.utils.signals import total_entropy, pca_analysis


def get_data(develop_dataset, label_selected, reservoir_model):
    # Scan through the time series while find a time series with the specified label
    i = 0
    u, label = develop_dataset[i]
    while label != label_selected:
        i = i + 1
        u, label = develop_dataset[i]  # u has shape (H=1, L)

    u_t = u.squeeze(0)  # (L,)

    y = reservoir_model(u.unsqueeze(0).to(device=check_model_device(reservoir_model)))  # (B=1, H=num_layers, L)
    y_t = y.squeeze(0)  # (H=num_layers, L)

    return u_t, y_t, label  # (L,), (H=num_layers, L), int


def plot_pca(y_t, label, save_path):
    n_layers = y_t.shape[0]  # Use the number of features

    fig, ax = plt.subplots(figsize=(14, 4))

    loadings = pca_analysis(y_t.t())  # take (num_samples=L, num_features=n_layers)

    width = 0.4  # Width of the bars
    k = np.arange(start=1, stop=n_layers+1)

    # Plotting the loadings for the first principal component
    ax.bar(k, loadings[0], width, label='PCA Loadings', color='green')

    # Set custom x-axis labels
    ax.set_xticks(k)
    ax.set_xticklabels(k)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Loading')
    ax.set_title(f'PCA Loadings of Output Signals (Label: {label})')
    ax.legend()

    # Save plot to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directories if not exist
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', default='cuda:3', help='Cuda device.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--label', type=int, default=0, help='Label to highlight in the time series plot.')
    parser.add_argument('--block', choices=['RSSM', 'ESN'], default='RSSM',
                        help='Block class to use for the model.')
    parser.add_argument('--layers', type=int, default=16, help='Number of layers.')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block == 'ESN':
        parser.add_argument('--inputscaling', type=float, default=1.0, help='Scaling of input matrix.')
        parser.add_argument('--biasscaling', type=float, default=0.0, help='Scaling of input matrix.')
        parser.add_argument('--rho', type=float, default=1.0, help='Spectral Radius of hidden state matrix.')
        parser.add_argument('--leaky', type=float, default=1.0, help='Leakage Rate for leaky integrator.')
    elif args.block == 'RSSM':
        parser.add_argument('--dstate', type=int, default=64, help='State size.')
        parser.add_argument('--minscaleencoder', type=float, default=0.0, help='Min encoder model scaling factor.')
        parser.add_argument('--maxscaleencoder', type=float, default=1.0, help='Max encoder model scaling factor.')
        parser.add_argument('--minscaleD', type=float, default=0.0, help='Skip connection matrix D min scaling.')
        parser.add_argument('--maxscaleD', type=float, default=1.0, help='Skip connection matrix D max scaling.')
        parser.add_argument('--kernel', choices=['Vr', 'miniVr'], default='Vr',
                            help='Kernel name.')
        parser.add_argument('--funfwd', default='real+relu',
                            help='Real function of complex variable to the Forward Pass.')
        parser.add_argument('--funfit', default='real+tanh',
                            help='Real function of complex variable to Fit the Readout.')
        parser.add_argument('--strong', type=float, default=-1.0, help='Strong Stability for internal dynamics.')
        parser.add_argument('--weak', type=float, default=0.0, help='Weak Stability for internal dynamics.')
        parser.add_argument('--discrete', action='store_true', help='Discrete SSM modality.')
        parser.add_argument('--low', type=float, default=0.001,
                            help='Min-Sampling-Rate / Min-Oscillations for internal dynamics.')
        parser.add_argument('--high', type=float, default=0.1,
                            help='Max-Sampling-Rate / Max-Oscillations for internal dynamics.')
        parser.add_argument('--minscaleB', type=float, default=0.0, help='Min scaling for input2state matrix B.')
        parser.add_argument('--maxscaleB', type=float, default=1.0, help='Max scaling for input2state matrix B.')
        parser.add_argument('--minscaleC', type=float, default=0.0, help='Min scaling for state2output matrix C.')
        parser.add_argument('--maxscaleC', type=float, default=1.0, help='Max scaling for state2output matrix C.')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logging.info(f"Setting seed: {args.seed}")
    set_seed(args.seed)

    if args.task in ['smnist', 'pmnist']:
        d_input = 1  # number of input features
        kernel_size = 28 * 28  # max length of input sequence
        label_list = list(range(10))
    elif args.task == 'pathfinder':
        d_input = 1
        kernel_size = 32 * 32
        label_list = list(range(2))
    elif args.task == 'pathx':
        d_input = 1
        kernel_size = 128 * 128
        label_list = list(range(2))
    elif args.task == 'scifar10gs':
        d_input = 1
        kernel_size = 32 * 32
        label_list = list(range(10))
    else:
        raise ValueError('Invalid task name')

    if args.label not in label_list:
        raise ValueError(f'Invalid label: {args.label} for task: {args.task}. Possible labels: {label_list}')

    if args.block == 'ESN':
        block_args = {'input_scaling': args.inputscaling, 'bias_scaling': args.biasscaling,
                      'spectral_radius': args.rho, 'leakage_rate': args.leaky}
    elif args.block == 'RSSM':
        block_args = {'fun_forward': args.funfwd,
                      'fun_fit': args.funfit,
                      'min_scaleD': args.minscaleD,
                      'max_scaleD': args.maxscaleD,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'strong_stability': args.strong, 'weak_stability': args.weak,
                      'discrete': args.discrete,
                      'low_oscillation': args.low, 'high_oscillation': args.high,
                      'min_scaleB': args.minscaleB,
                      'max_scaleB': args.maxscaleB,
                      'min_scaleC': args.minscaleC,
                      'max_scaleC': args.maxscaleC}
    else:
        raise ValueError('Invalid block name')

    save_path = os.path.join('./checkpoint', 'dynamics', args.task, args.block, 'pca.png')

    logging.info('Loading develop dataset.')
    develop_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))

    logging.info('Initializing model.')
    if args.block == 'RSSM':
        reservoir_model = StackedReservoir(block_cls=RSSM,
                                           n_layers=args.layers,
                                           d_input=d_input, d_model=1, d_state=args.dstate,
                                           transient=0,
                                           take_last=False,
                                           min_encoder_scaling=args.minscaleencoder,
                                           max_encoder_scaling=args.maxscaleencoder,
                                           **block_args)
        logging.info(f'Moving reservoir model to {args.device}.')
        reservoir_model.to(device=torch.device(args.device))

    elif args.block == 'ESN':
        reservoir_model = StackedEchoState(n_layers=args.layers,
                                           d_input=d_input, d_model=1,
                                           transient=0,
                                           take_last=False,
                                           **block_args)
        logging.info(f'Moving reservoir model to {args.device}.')
        reservoir_model.to(device=torch.device(args.device))

    else:
        raise ValueError('Invalid block name')

    logging.info('Retrieve data.')
    u_t, y_t, label = get_data(develop_dataset, args.label, reservoir_model)  # (L,), (H=num_layers, L), int

    logging.info('Plotting.')
    plot_pca(y_t, label, save_path)


if __name__ == '__main__':
    main()