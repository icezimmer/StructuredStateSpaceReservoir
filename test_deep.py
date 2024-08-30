import argparse
import logging
import os
import torch
import matplotlib.pyplot as plt

from src.deep.stacked import StackedReservoir, StackedEchoState
from src.models.rssm.rssm import RSSM
from src.utils.experiments import set_seed
from src.utils.saving import load_data
from src.utils.experiments import read_yaml_to_dict


def get_data(develop_dataset, label_selected, reservoir_model):
    # Scan through the time series while find a time series with the specified label
    i = 0
    item = develop_dataset[i]
    u, label = item[0], item[1]
    if len(item) == 3:
        length = item[2]
        u = u[..., :length]
    while label != label_selected:
        i = i + 1
        item = develop_dataset[i]  # u has shape (H0, L)
        u, label = item[0], item[1]
        if len(item) == 3:
            length = item[2]
            u = u[..., :length]

    v = reservoir_model.encoder(u.unsqueeze(0))  # (B=1, H=1, L)
    v_t = v.squeeze()  # (L,)

    y = reservoir_model(u.unsqueeze(0))  # (B=1, H=num_layers, L)
    y_t = y.squeeze(0)  # (H=num_layers, L)

    return v_t, y_t, label  # (L,), (H=num_layers, L), int


def plot_time_series(v_t, y_t, label, save_path):
    n_layers = y_t.shape[0]
    length = v_t.shape[-1]

    fig = plt.figure(figsize=(20, 4*(n_layers+1)))

    fig_v = fig.add_subplot(n_layers+1, 2, 2*(n_layers+1)-1)
    v_np = v_t.cpu().numpy()
    fig_v.plot(range(length), v_np)
    fig_v.set_ylabel('Encoding', fontsize=22)

    # Compute the DFT of the time series
    v_s = torch.fft.rfft(v_t, n=2*length-1, dim=-1)
    freq = torch.fft.rfftfreq(n=2*length-1)
    # Compute the amplitude of the DFT
    amplitude = torch.abs(v_s).cpu().numpy()
    fig_v_s = fig.add_subplot(n_layers + 1, 2, 2*(n_layers+1))
    fig_v_s.bar(freq, amplitude, width=0.01)

    for i in range(n_layers):
        fig_h = fig.add_subplot(n_layers+1, 2, 2*(n_layers-i)-1)
        h_t = y_t[i, :]  # h has shape (L,)
        h_np = h_t.cpu().numpy()
        fig_h.plot(range(length), h_np)
        fig_h.set_ylabel(f'Layer {i+1}', fontsize=22)
        if i == n_layers - 1:
            fig_h.set_title('Output signal', fontsize=22)

        # Compute the DFT of the time series
        h_s = torch.fft.rfft(h_t, n=2*length - 1, dim=-1)
        freq = torch.fft.rfftfreq(n=2 * length - 1)
        # Compute the amplitude of the DFT
        amplitude = torch.abs(h_s).cpu().numpy()
        fig_h_s = fig.add_subplot(n_layers + 1, 2, 2*(n_layers-i))
        fig_h_s.bar(freq, amplitude, width=0.01)
        if i == n_layers - 1:
            fig_h_s.set_title(f'Frequency amplitude', fontsize=22)

    # Save plot to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directories if not exist
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification task.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--task', default='smnist', help='Name of task.')
    parser.add_argument('--label', type=int, default=0, help='Label to highlight in the time series plot.')
    parser.add_argument('--block', choices=['RSSM', 'ESN'], default='RSSM',
                        help='Block class to use for the model.')
    parser.add_argument('--layers', type=int, default=6, help='Number of layers.')

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
        parser.add_argument('--low', type=float, default=0.001, help='Min-Sampling-Rate / Min-Oscillations for internal dynamics.')
        parser.add_argument('--high', type=float, default=0.1, help='Max-Sampling-Rate / Max-Oscillations for internal dynamics.')
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

    setting = read_yaml_to_dict(os.path.join('configs', args.task, 'setting.yaml'))
    architecture = setting.get('architecture', {})
    d_input = architecture['d_input']  # dim of input space or vocab size for text embedding
    kernel_size = architecture['kernel_size']
    label_list = list(range(architecture['d_output']))

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

    save_path = os.path.join('./checkpoint', 'dynamics', args.task, args.block, 'deep.pdf')

    logging.info('Loading develop dataset.')
    develop_dataset = load_data(os.path.join('..', 'datasets', args.task, 'develop_dataset'))

    logging.info('Initializing model.')
    if args.block == 'RSSM':
        reservoir_model = StackedReservoir(block_cls=RSSM,
                                           n_layers=args.layers,
                                           d_input=d_input, d_model=1, d_state=args.dstate,
                                           transient=0,
                                           take_last=False,
                                           encoder='onehot' if args.task in ['listops', 'imdb'] else 'reservoir',
                                           min_encoder_scaling=args.minscaleencoder,
                                           max_encoder_scaling=args.maxscaleencoder,
                                           **block_args)

    elif args.block == 'ESN':
        reservoir_model = StackedEchoState(n_layers=args.layers,
                                           d_input=d_input, d_model=1,
                                           transient=0,
                                           take_last=False,
                                           **block_args)

    else:
        raise ValueError('Invalid block name')

    logging.info('Retrieve data.')
    v_t, y_t, label = get_data(develop_dataset, args.label, reservoir_model)  # (L,), (H=num_layers, L), int

    logging.info('Plotting.')
    plot_time_series(v_t, y_t, label, save_path)


if __name__ == '__main__':
    main()