import argparse
import logging
import os
from sklearn.linear_model import RidgeClassifierCV
import torch
from src.utils.experiments import set_seed
from torch.utils.data import DataLoader
from src.utils.split_data import stratified_split_dataset
from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla import VanillaRNN, VanillaGRU, VanillaLSTM
from src.models.esn.esn import ESN
from src.models.s4d.s4d import S4D
from src.models.lrssm.lrssm import LRSSM
from src.models.rssm.rssm import RSSM
from src.deep.stacked import StackedNetwork, StackedReservoir, StackedEchoState
from src.readout.mlp import MLP
from src.readout.ridge import RidgeRegression
from src.torch_dataset.reservoir_to_readout import Reservoir2ReadOut
from src.ml.optimization import setup_optimizer
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier, EvaluateOfflineClassifier
from src.utils.saving import load_data, save_hyperparameters, update_results, update_hyperparameters
from src.utils.check_device import check_model_device
from src.utils.experiments import read_yaml_to_dict
from codecarbon import EmissionsTracker

block_factories = {
    'S4': S4Block,
    'RNN': VanillaRNN,
    'GRU': VanillaGRU,
    'LSTM': VanillaLSTM,
    'S4D': S4D,
    'LRSSM': LRSSM,
    'ESN': ESN,
    'RSSM': RSSM
}

loss = {
    'cross_entropy': torch.nn.CrossEntropyLoss()
}

conv_classes = ['fft', 'fft-freezeD']
kernel_classes = ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC', 'V-freezeABC']

lrssm_activations = ['relu', 'tanh', 'glu']

kernel_classes_reservoir = ['Vr']
realfuns = ['real', 'real+relu', 'real+tanh', 'realimag+glu', 'abs+tanh', 'angle+tanh']
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
    parser.add_argument('--dmodel', type=int, default=64, help='Dimension of each hidden layer.')

    # First parse known arguments to decide on adding additional arguments based on the block type
    args, unknown = parser.parse_known_args()

    # Conditional argument additions based on block type
    if args.block in ['RNN', 'GRU', 'LSTM', 'S4', 'S4D', 'LRSSM']:
        parser.add_argument('--batch', type=int, default=128, help='Batch size')
        parser.add_argument('--encoder', default='conv1d', help='Encoder model.')
        parser.add_argument('--decoder', default='conv1d', help='Decoder model.')
        parser.add_argument('--layerdrop', type=float, default=0.0, help='Dropout the output of each layer.')
        parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for NON-kernel parameters.')
        parser.add_argument('--wd', type=float, default=0.01, help='Weight decay for NON-kernel parameters.')
        parser.add_argument('--plateau', type=float, default=0.2, help='Learning rate decay factor on Plateau.')
        parser.add_argument('--epochs', type=int, default=float('inf'), help='Number of epochs.')
        parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping.')
        if args.block in ['RNN', 'GRU', 'LSTM']:
            pass
        elif args.block == 'S4':
            parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
            parser.add_argument('--tiedropout', action='store_true', help='Tie dropout.')
            parser.add_argument('--dstate', type=int, default=64, help='State size.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--low', type=float, default=0.001, help='Min-Sampling-Rate for internal dynamics.')
            parser.add_argument('--high', type=float, default=0.1, help='Max-Sampling-Rate for internal dynamics.')
            parser.add_argument('--init', default='legs', help='Choices for initialization of A')
            parser.add_argument('--bidirectional', action='store_true', help='Bidirectional.')
            parser.add_argument('--finalact', default='glu', help='Activation.')
            parser.add_argument('--nssm', type=int, default=1, help='Kernel name.')            
        elif args.block == 'S4D':
            parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
            parser.add_argument('--conv', choices=conv_classes, default='fft', help='Skip connection matrix D.')
            parser.add_argument('--minscaleD', type=float, default=0.0, help='Skip connection matrix D min scaling.')
            parser.add_argument('--maxscaleD', type=float, default=1.0, help='Skip connection matrix D max scaling.')
            parser.add_argument('--kerneldrop', type=float, default=0.0, help='Dropout the kernel inside the block.')
            parser.add_argument('--kernel', choices=kernel_classes, default='V', help='Kernel name.')
            parser.add_argument('--mix', default='conv1d+glu', help='Inner Mixing layer.')
            parser.add_argument('--strong', type=float, default=-1.0, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=0.0, help='Weak Stability for internal dynamics.')
            parser.add_argument('--kernellr', type=float, default=0.001, help='Learning rate for kernel pars.')
            parser.add_argument('--kernelwd', type=float, default=0.0, help='Learning rate for kernel pars.')
            parser.add_argument('--dt', type=float, default=0.05, help='Sampling rate (only for continuous dynamics).')
            parser.add_argument('--minscaleB', type=float, default=0.0, help='Min scaling for input2state matrix B.')
            parser.add_argument('--maxscaleB', type=float, default=1.0, help='Max scaling for input2state matrix B.')
            parser.add_argument('--minscaleC', type=float, default=0.0, help='Min scaling for state2output matrix C.')
            parser.add_argument('--maxscaleC', type=float, default=1.0, help='Max scaling for state2output matrix C.')
        elif args.block == 'LRSSM':
            parser.add_argument('--dstate', type=int, default=64, help='State size.')
            parser.add_argument('--dropout', type=float, default=0.0, help='Dropout the preactivation inside the block.')
            parser.add_argument('--minscaleD', type=float, default=0.0, help='Skip connection matrix D min scaling.')
            parser.add_argument('--maxscaleD', type=float, default=1.0, help='Skip connection matrix D max scaling.')
            parser.add_argument('--kernel', choices=kernel_classes_reservoir, default='Vr', help='Kernel name.')
            parser.add_argument('--act', choices=lrssm_activations, default='glu', help='Kernel name.')
            parser.add_argument('--strong', type=float, default=-1.0, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=0.0, help='Weak Stability for internal dynamics.')
            parser.add_argument('--low', type=float, default=0.001, help='Min-Sampling-Rate / Min-Oscillations for internal dynamics.')
            parser.add_argument('--high', type=float, default=0.1, help='Max-Sampling-Rate / Max-Oscillations for internal dynamics.')
            parser.add_argument('--minscaleB', type=float, default=0.0, help='Min scaling for input2state matrix B.')
            parser.add_argument('--maxscaleB', type=float, default=1.0, help='Max scaling for input2state matrix B.')
            parser.add_argument('--minscaleC', type=float, default=0.0, help='Min scaling for state2output matrix C.')
            parser.add_argument('--maxscaleC', type=float, default=1.0, help='Max scaling for state2output matrix C.')
            parser.add_argument('--mlplayers', type=int, default=1, help='Number of MLP layers.')
    elif args.block in ['ESN', 'RSSM']:
        parser.add_argument('--rbatch', type=int, default=128, help='Batch size for Reservoir Model.')
        parser.add_argument('--last', action='store_true', help='Take only the last layer output.')
        parser.add_argument('--readout', choices=readout_classes, default='ridge', help='Type of Readout.')
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
            parser.add_argument('--kernel', choices=kernel_classes_reservoir, default='Vr',
                                help='Kernel name.')
            parser.add_argument('--funfwd', choices=realfuns, default='real+relu', help='Real function of complex variable to the Forward Pass.')
            parser.add_argument('--funfit', choices=realfuns, default='real+tanh', help='Real function of complex variable to Fit the Readout.')
            parser.add_argument('--strong', type=float, default=-1.0, help='Strong Stability for internal dynamics.')
            parser.add_argument('--weak', type=float, default=0.0, help='Weak Stability for internal dynamics.')
            parser.add_argument('--discrete', action='store_true', help='Discrete SSM modality.')
            parser.add_argument('--low', type=float, default=0.001, help='Min-Sampling-Rate / Min-Oscillations for internal dynamics.')
            parser.add_argument('--high', type=float, default=0.1, help='Max-Sampling-Rate / Max-Oscillations for internal dynamics.')
            parser.add_argument('--minscaleB', type=float, default=0.0, help='Min scaling for input2state matrix B.')
            parser.add_argument('--maxscaleB', type=float, default=1.0, help='Max scaling for input2state matrix B.')
            parser.add_argument('--minscaleC', type=float, default=0.0, help='Min scaling for state2output matrix C.')
            parser.add_argument('--maxscaleC', type=float, default=1.0, help='Max scaling for state2output matrix C.')

    # Update args with the new conditional arguments
    args, unknown = parser.parse_known_args()

    if hasattr(args, 'encoder'):
        parser.add_argument('--minscaleencoder', type=float, default=0.0, help='Min encoder model scaling factor.')
        parser.add_argument('--maxscaleencoder', type=float, default=1.0, help='Max encoder model scaling factor.')

    if hasattr(args, 'decoder'):
        parser.add_argument('--minscaledecoder', type=float, default=0.0, help='Min decoder model scaling factor.')
        parser.add_argument('--maxscaledecoder', type=float, default=1.0, help='Max decoder model scaling factor.')

    if hasattr(args, 'readout'):
        if args.readout == 'ridge':
            parser.add_argument('--transient', type=int, default=-1, help='Number of first time steps to discard.')
            parser.add_argument('--regul', type=float, default=0.8, help='Regularization for Ridge Regression.')

        if args.readout == 'mlp':
            parser.add_argument('--batch', type=int, default=128, help='Batch size')
            parser.add_argument('--transient', type=int, default=-1, help='Number of first time steps to discard.')
            parser.add_argument('--mlplayers', type=int, default=2, help='Number of MLP layers.')
            parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for MLP parameters.')
            parser.add_argument('--wd', type=float, default=0.01, help='Weight decay for MLP parameters.')
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

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)

    # Check if cuDNN is enabled
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    torch.cuda.empty_cache()
    # smaller -> less memory, slower process | larger -> more memory, faster process
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    args = parse_args()

    logging.info(f"Setting seed: {args.seed}")
    set_seed(args.seed)

    setting = read_yaml_to_dict(os.path.join('configs', args.task, 'setting.yaml'))
    architecture = setting.get('architecture', {})
    criterion = loss[architecture['criterion']]
    to_vec = architecture['to_vec']
    d_input = architecture['d_input']  # dim of input space or vocab size for text embedding
    kernel_size = architecture['kernel_size']
    d_output = architecture['d_output']
    mode = setting.get('mode', "")

    learning = setting.get('learning', {})
    val_split = learning.get('val_split')

    if args.block in ['RNN', 'GRU', 'LSTM']:
        block_args = {}
    elif args.block == 'S4':
        block_args = {'dropout': args.dropout, 'tie_dropout': args.tiedropout,
                      'd_state': args.dstate, 'lr': args.kernellr, 'dt_min': args.low, 'dt_max': args.high, 'init': args.init,
                      'bidirectional': args.bidirectional, 'final_act': args.finalact, 'n_ssm': args.nssm}
    elif args.block == 'S4D':
        block_args = {'mixing_layer': args.mix,
                      'convolution': args.conv,
                      'min_scaleD': args.minscaleD,
                      'max_scaleD': args.maxscaleD,
                      'drop_kernel': args.kerneldrop, 'dropout': args.dropout,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'strong_stability': args.strong, 'weak_stability': args.weak,
                      'dt': args.dt,
                      'min_scaleB': args.minscaleB,
                      'max_scaleB': args.maxscaleB,
                      'min_scaleC': args.minscaleC,
                      'max_scaleC': args.maxscaleC}
        if args.kernel in ['V', 'V-freezeB', 'V-freezeC', 'V-freezeBC', 'V-freezeA', 'V-freezeAB', 'V-freezeAC']:
            block_args['lr'] = args.kernellr
            block_args['wd'] = args.kernelwd
    elif args.block == 'LRSSM':
        block_args = {'d_state': args.dstate,
                      'min_scaleD': args.minscaleD,
                      'max_scaleD': args.maxscaleD,
                      'dropout': args.dropout,
                      'kernel': args.kernel, 'kernel_size': kernel_size,
                      'act': args.act,
                      'strong_stability': args.strong, 'weak_stability': args.weak,
                      'low_oscillation': args.low, 'high_oscillation': args.high,
                      'min_scaleB': args.minscaleB,
                      'max_scaleB': args.maxscaleB,
                      'min_scaleC': args.minscaleC,
                      'max_scaleC': args.maxscaleC,
                      'mlp_layers': args.mlplayers}
    elif args.block == 'ESN':
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

    if args.block == 'S4D':
        block_name = args.block + '_' + args.conv + '_' + args.kernel + '_' + args.mix
    elif args.block == 'LRSSM':
        block_name = args.block + '_' + args.kernel + '_conv1d' + '_' + args.act
    elif args.block == 'RSSM':
        block_name = args.block + '_' + args.kernel + '_^' + args.funfwd + '_>' + args.funfit
    else:
        block_name = args.block

    if args.block not in ['ESN', 'RSSM']:
        project_name = (args.encoder + '_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.dmodel) + ']_' +
                        args.decoder)
    elif args.block == 'RSSM':
        project_name = ('reservoir_[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.dmodel) + ']_' +
                        args.readout)
    elif args.block == 'ESN':
        project_name = ('[{' + block_name + '}_' + str(args.layers) + 'x' + str(args.dmodel) + ']_' +
                        args.readout)
    else:
        raise ValueError('Invalid block name')

    output_dir = os.path.join('./checkpoint', 'results', args.task)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f'Loading {args.task} develop and test datasets.')
    try:
        develop_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'develop_dataset'))
        test_dataset = load_data(os.path.join('./checkpoint', 'datasets', args.task, 'test_dataset'))
    except FileNotFoundError:
        logging.error(f"Dataset not found for task {args.task}. Run build.py first.")

    if args.block in ['RNN', 'GRU', 'LSTM', 'S4', 'S4D', 'LRSSM']:
        log_file_name = args.block

        develop_dataloader = DataLoader(develop_dataset,
                                        batch_size=args.batch,
                                        shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch,
                                     shuffle=False)

        logging.info(f'Initializing {args.block} model.')
        model = StackedNetwork(block_cls=block_factories[args.block], n_layers=args.layers,
                               d_input=d_input, d_model=args.dmodel, d_output=d_output,
                               encoder=args.encoder, decoder=args.decoder,
                               to_vec=to_vec,
                               min_encoder_scaling=args.minscaleencoder, max_encoder_scaling=args.maxscaleencoder,
                               min_decoder_scaling=args.minscaledecoder, max_decoder_scaling=args.maxscaledecoder,
                               layer_dropout=args.layerdrop,
                               **block_args)

        logging.info(f'Moving {args.block} model to {args.device}.')
        model.to(device=torch.device(args.device))

        logging.info('Splitting develop data into training and validation data.')
        train_dataset, val_dataset = stratified_split_dataset(dataset=develop_dataset, val_split=val_split)
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

        logging.info(f'[Tracking] Fitting {args.block} model for {args.task}.')
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

        if mode == 'classification':
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

        develop_dataloader = DataLoader(develop_dataset,
                                        batch_size=args.rbatch,
                                        shuffle=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.rbatch,
                                     shuffle=False)

        logging.info(f'Initializing {args.block} model.')
        if args.block == 'RSSM':
            reservoir_model = StackedReservoir(block_cls=block_factories[args.block],
                                               n_layers=args.layers,
                                               d_input=d_input, d_model=args.dmodel, d_state=args.dstate,
                                               transient=args.transient,
                                               take_last=args.last,
                                               min_encoder_scaling=args.minscaleencoder,
                                               max_encoder_scaling=args.maxscaleencoder,
                                               **block_args)
            logging.info(f'Moving {args.block} model to {args.device}.')
            reservoir_model.to(device=torch.device(args.device))

        elif args.block == 'ESN':
            reservoir_model = StackedEchoState(n_layers=args.layers,
                                               d_input=d_input, d_model=args.dmodel,
                                               transient=args.transient,
                                               take_last=args.last,
                                               **block_args)
            logging.info(f'Moving {args.block} model to {args.device}.')
            reservoir_model.to(device=torch.device(args.device))

        else:
            raise ValueError('Invalid block name')

        if args.readout == 'ridge':
            model = RidgeRegression(d_input=reservoir_model.d_output, d_output=d_output, alpha=args.regul, to_vec=to_vec)

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

            logging.info(f'[Tracking] Fitting {args.block} model for {args.task}.')
            tracker.start()
            develop_dataset = Reservoir2ReadOut(reservoir_model=reservoir_model, dataloader=develop_dataloader)
            X, y = develop_dataset.to_fit_offline_readout()
            _ = model(X, y)

            # clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X.numpy(), y.numpy())
            # print(clf.score(X.numpy(), y.numpy()))

            emissions = tracker.stop()
            logging.info(f"Estimated CO2 emissions for this fit: {emissions} kg")

            if args.save:
                logging.info('Saving model hyper-parameters.')
                save_hyperparameters(args=args, file_path=hyperparameters_path)
                logging.info('Saving reservoir model.')
                torch.save(reservoir_model.state_dict(), reservoir_model_path)

            if mode == 'classification':
                if args.tr:
                    logging.info('Evaluating model on develop set.')
                    X, y = develop_dataset.to_evaluate_offline_classifier()
                    eval_dev = EvaluateOfflineClassifier()
                    eval_dev.evaluate(y_true=y.numpy(), y_pred=model(X).numpy())

                logging.info('Evaluating model on test set.')
                test_dataset = Reservoir2ReadOut(reservoir_model=reservoir_model, dataloader=test_dataloader)
                X, y = test_dataset.to_evaluate_offline_classifier()
                eval_test = EvaluateOfflineClassifier()
                eval_test.evaluate(y_true=y.numpy(), y_pred=model(X).numpy())
                scores = {'test_accuracy': eval_test.accuracy_value}
            else:
                scores = {}

        elif args.readout == 'mlp':
            model = MLP(n_layers=args.mlplayers, d_input=reservoir_model.d_output, d_output=d_output)

            logging.info(f'Moving {args.block} model to {args.device}.')
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

            logging.info(f'[Tracking] Fitting {args.block} model for {args.task}.')
            tracker.start()
            develop_dataset = Reservoir2ReadOut(reservoir_model=reservoir_model, dataloader=develop_dataloader)
            develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=False)

            train_dataset, val_dataset = stratified_split_dataset(dataset=develop_dataset, val_split=val_split)
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

            if mode == 'classification':
                if args.tr:
                    logging.info('Evaluating model on develop set.')
                    eval_dev = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                    eval_dev.evaluate(saving_path=develop_path)

                logging.info(f'Computing reservoir test set.')
                test_dataset = Reservoir2ReadOut(reservoir_model=reservoir_model, dataloader=test_dataloader)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

                logging.info('Evaluating model on test set.')
                eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
                eval_test.evaluate(saving_path=test_path)
                scores = {'test_accuracy': eval_test.accuracy_value}
            else:
                scores = {}

        elif args.readout == 'ssm':
            model = StackedNetwork(block_cls=S4Block, n_layers=args.ssmlayers,
                                   d_input=reservoir_model.d_output, d_model=reservoir_model.d_output,
                                   d_output=d_output,
                                   encoder='conv1d', decoder='conv1d',
                                   to_vec=to_vec)

            logging.info(f'Moving {args.block} model to {args.device}.')
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

            logging.info(f'[Tracking] Fitting {args.block} model for {args.task}.')
            tracker.start()
            develop_dataset = Reservoir2ReadOut(reservoir_model=reservoir_model, dataloader=develop_dataloader)
            develop_dataloader = DataLoader(develop_dataset, batch_size=args.batch, shuffle=False)

            train_dataset, val_dataset = stratified_split_dataset(dataset=develop_dataset, val_split=val_split)
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

            if mode == 'classification':
                if args.tr:
                    logging.info('Evaluating model on develop set.')
                    eval_dev = EvaluateClassifier(model=model, num_classes=d_output, dataloader=develop_dataloader)
                    eval_dev.evaluate(saving_path=develop_path)

                logging.info(f'Computing reservoir test set.')
                test_dataset = Reservoir2ReadOut(reservoir_model=reservoir_model, dataloader=test_dataloader)
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
