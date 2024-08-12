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


def plot_entropy(u_t, y_t, label, save_path):

    n_layers = y_t.shape[0]

    fig = plt.figure(figsize=(14, 4))
    fig_ts = fig.add_subplot(1, 2, 1)
    fig_x = fig.add_subplot(1, 2, 2)

    entropy_cumulative = np.array([0])
    entropy_cumulative_x = np.array([0])
    for i in range(n_layers):
        h_t = y_t[0:(i + 1), :]  # h has shape (H=i+1, L)
        entropy = total_entropy(h_t, dim=h_t.shape[0])
        entropy_np = entropy.unsqueeze(0).cpu().numpy()
        entropy_cumulative = np.concatenate((entropy_cumulative, entropy_np))
        x = h_t[:, -1]
        entropy_x = total_entropy(x, dim=x.shape[0])
        entropy_x_np = entropy_x.unsqueeze(0).cpu().numpy()
        entropy_cumulative_x = np.concatenate((entropy_cumulative_x, entropy_x_np))

    entropy_u = total_entropy(u_t, dim=1)
    entropy_single = entropy_u.unsqueeze(0).cpu().numpy()
    for i in range(n_layers):
        h_t = y_t[i:(i + 1), :]  # h has shape (H=1, L)
        entropy = total_entropy(h_t, dim=1)
        entropy_np = entropy.unsqueeze(0).cpu().numpy()
        entropy_single = np.concatenate((entropy_single, entropy_np))

    width = 0.4  # Width of the bars
    k = np.arange(n_layers + 1)
    fig_ts.bar(k - width / 2, entropy_single, width, label='Single', color='red')
    fig_ts.bar(k + width / 2, entropy_cumulative, width, label='Aggregated', color='green')
    fig_x.bar(k, entropy_cumulative_x, width, label='Aggregated', color='green')

    # Set custom x-axis labels
    fig_ts.set_xticks(k)
    fig_ts.set_xticklabels(k)
    fig_x.set_xticks(k)
    fig_x.set_xticklabels(k)

    fig_ts.set_xlabel('Layer')
    fig_ts.set_ylabel('Entropy')
    fig_ts.set_title(f'Entropy of Hidden Signals (Label: {label})')
    fig_ts.legend()

    fig_x.set_xlabel('Layer')
    fig_x.set_ylabel('Entropy')
    fig_x.set_title(f'Entropy of Hidden State (Label: {label})')
    fig_x.legend()

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
        parser.add_argument('--minscaleencoder', type=float, default=0.0, help='Min encoder model scaling factor.')
        parser.add_argument('--maxscaleencoder', type=float, default=1.0, help='Max encoder model scaling factor.')
        parser.add_argument('--minscaleD', type=float, default=0.0, help='Skip connection matrix D min scaling.')
        parser.add_argument('--maxscaleD', type=float, default=1.0, help='Skip connection matrix D max scaling.')
        parser.add_argument('--kernel', choices=['Vr', 'miniVr'], default='Vr',
                            help='Kernel name.')
        parser.add_argument('--funfwd', default='real',
                            help='Real function of complex variable to the Forward Pass.')
        parser.add_argument('--funfit', default='real',
                            help='Real function of complex variable to Fit the Readout.')
        parser.add_argument('--strong', type=float, default=-1.0, help='Strong Stability for internal dynamics.')
        parser.add_argument('--weak', type=float, default=0.0, help='Weak Stability for internal dynamics.')
        parser.add_argument('--discrete', action='store_true', help='Discrete SSM modality.')
        parser.add_argument('--dt', type=float, default=0.01, help='Sampling-Rate / Oscillations for internal dynamics.')
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
    elif args.task in ['pathfinder32easy', 'pathfinder32intermediate', 'pathfinder32hard']:
        d_input = 1
        kernel_size = 32 * 32
        label_list = list(range(2))
    elif args.task in ['pathfinder64easy', 'pathfinder64intermediate', 'pathfinder64hard']:
        d_input = 1
        kernel_size = 64 * 64
        label_list = list(range(2))
    elif args.task in ['pathfinder128easy', 'pathfinder128intermediate', 'pathfinder128hard']:
        d_input = 1
        kernel_size = 128 * 128
        label_list = list(range(2))
    elif args.task in ['pathfinder256easy', 'pathfinder256intermediate', 'pathfinder256hard']:
        d_input = 1
        kernel_size = 256 * 256
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
                      'low_oscillation': args.dt, 'high_oscillation': args.dt,
                      'min_scaleB': args.minscaleB,
                      'max_scaleB': args.maxscaleB,
                      'min_scaleC': args.minscaleC,
                      'max_scaleC': args.maxscaleC}
    else:
        raise ValueError('Invalid block name')

    save_path = os.path.join('./checkpoint', 'dynamics', args.task, args.block, 'entropy.png')

    logging.info('Loading develop dataset.')
    develop_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))

    logging.info('Initializing model.')
    if args.block == 'RSSM':
        reservoir_model = StackedReservoir(block_cls=RSSM,
                                           n_layers=args.layers,
                                           d_input=d_input, d_model=1,
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
    plot_entropy(u_t, y_t, label, save_path)


if __name__ == '__main__':
    main()