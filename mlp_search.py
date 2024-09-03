import argparse
import os
from src.utils.saving import load_data
import logging
import random
import itertools
from src.utils.experiments import read_yaml_to_dict
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier
import torch
from src.readout.mlp import MLP
from torch.utils.data import DataLoader
from src.utils.split_data import stratified_split_dataset
from src.utils.saving import save_hyperparameters


loss = {
    'cross_entropy': torch.nn.CrossEntropyLoss()
}


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run readout based on YAML configuration.")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to run the experiment.")
    parser.add_argument("--task", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--trials", type=int, required=True, help="Number of configurations to sample.")

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    logging.info(f'Loading {args.task} reservoir develop and test datasets.')
    try:
        develop_dataset = load_data(os.path.join('..', 'datasets', args.task, 'reservoir_develop_dataset'))
        test_dataset = load_data(os.path.join('..', 'datasets', args.task, 'reservoir_test_dataset'))
    except FileNotFoundError:
        logging.error(f"Dataset not found for task {args.task}. Run train.py first with --save flag.")

    setting = read_yaml_to_dict(os.path.join('configs', args.task, 'setting.yaml'))
    architecture = setting.get('architecture', {})
    d_output = architecture['d_output']
    criterion = loss[architecture['criterion']]

    learning = setting.get('learning', {})
    val_split = learning.get('val_split')

    device = args.device
    d_input = develop_dataset[0][0].shape[-2]

    experiments = {
        'batch': [64],
        'mlplayers': [2, 4, 6],
        'lr': [0.0005, 0.001, 0.005, 0.01],
        'wd': [0.01, 0.05, 0.1, 0.5],
        'plateau': [0.2],
        'epochs': [200],
        'patience': [10]
    }

    # Get the keys and values from the dictionary
    keys = experiments.keys()
    values = experiments.values()

    # Generate all combinations
    combinations = list(itertools.product(*values))

    # Convert to list of dictionaries
    configurations = [dict(zip(keys, combination)) for combination in combinations]

    # Shuffle experiments
    random.shuffle(configurations)

    # Sample a subset of experiments
    sampled_experiments = configurations[:args.trials]

    # Generate and execute sampled experiments
    best_score = 0.0
    for experiment in sampled_experiments:
        print(experiment)

        model = MLP(n_layers=experiment['mlplayers'], d_input=d_input, d_output=d_output)

        logging.info(f'Moving MLP model to {device}.')
        model.to(device=torch.device(device))

        logging.info('Setting optimizer.')
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=experiment['lr'], weight_decay=experiment['wd'])

        develop_dataloader = DataLoader(develop_dataset, batch_size=experiment['batch'], shuffle=False)

        train_dataset, val_dataset = stratified_split_dataset(dataset=develop_dataset, val_split=val_split)
        train_dataloader = DataLoader(train_dataset, batch_size=experiment['batch'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=experiment['batch'], shuffle=False)

        trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                             develop_dataloader=develop_dataloader)

        logging.info('Training MLP readout.')
        trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               patience=experiment['patience'], num_epochs=experiment['epochs'],
                               reduce_plateau=experiment['plateau'],
                               plot_path=None)

        logging.info(f'Computing reservoir test set.')
        test_dataloader = DataLoader(test_dataset, batch_size=experiment['batch'], shuffle=False)

        logging.info('Evaluating model on test set.')
        eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
        eval_test.evaluate(saving_path=None)
        score = eval_test.accuracy_value

        if score > best_score:
            experiment['test_accuracy'] = score
            save_hyperparameters(dictionary=experiment,
                                 file_path=os.path.join('checkpoint', 'results', args.task, 'mlp', 'best.json'))
            best_score = score


if __name__ == "__main__":
    main()
