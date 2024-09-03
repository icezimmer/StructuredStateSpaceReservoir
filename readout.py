import argparse
import os
from src.utils.saving import load_data
import logging
import random
import numpy as np
from src.utils.experiments import read_yaml_to_dict
from src.readout.ridge import RidgeRegression
from src.ml.training import TrainModel
from src.ml.evaluation import EvaluateClassifier, EvaluateOfflineClassifier
import torch
from src.readout.mlp import MLP
from torch.utils.data import DataLoader
from src.utils.split_data import stratified_split_dataset
from sklearn.linear_model import RidgeClassifierCV
from src.utils.saving import save_hyperparameters


loss = {
    'cross_entropy': torch.nn.CrossEntropyLoss()
}


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run readout based on YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    # Read hyperparameters from the YAML file
    config = read_yaml_to_dict(args.config)

    max_configs = config.get('max_configs', 0)
    hyperparameters = config.get('hyperparameters', {})

    # Extract the keys and possible values for each hyperparameter
    keys, values = zip(*hyperparameters.items())

    # Ensure max_configs is not larger than the total possible combinations
    total_experiment_count = np.prod([len(v) for v in values])
    if max_configs > total_experiment_count:
        max_configs = total_experiment_count

    # Perform random sampling of hyperparameter combinations
    sampled_experiments = []
    for _ in range(max_configs):
        experiment = {key: random.choice(value) for key, value in zip(keys, values)}
        sampled_experiments.append(experiment)

    experiment = sampled_experiments[0]
    task = experiment['task']
    readout = experiment['readout']
    setting = read_yaml_to_dict(os.path.join('configs', task, 'setting.yaml'))
    architecture = setting.get('architecture', {})
    d_output = architecture['d_output']
    to_vec = architecture['to_vec']
    criterion = loss[architecture['criterion']]

    learning = setting.get('learning', {})
    val_split = learning.get('val_split')

    logging.info(f'Loading {task} reservoir develop and test datasets.')
    try:
        develop_dataset = load_data(os.path.join('..', 'datasets', task, 'reservoir_develop_dataset'))
        test_dataset = load_data(os.path.join('..', 'datasets', task, 'reservoir_test_dataset'))
    except FileNotFoundError:
        logging.error(f"Dataset not found for task {task}. Run train.py first with --save flag.")

    # Generate and execute sampled experiments
    best_score = 0.0
    for experiment in sampled_experiments:
        print(experiment)
        if readout == 'ridgeCV':
            X, y = develop_dataset.to_fit_offline_readout()
            logging.info('Ridge Classifier Cross Validation.')
            clf = RidgeClassifierCV(alphas=hyperparameters['regul'], cv=3).fit(X.numpy(), y.numpy())

            X, y = test_dataset.to_evaluate_offline_classifier()
            y_pred = clf.predict(X.numpy())
            score = clf.score(y_pred.numpy(), y.numpy())
            print(score)
            experiment['test_accuracy'] = score
            save_hyperparameters(dictionary=experiment, file_path=os.path.join('checkpoint', 'result', 'ridge.json'))
            break

        elif readout == 'ridge':
            d_input = experiment['dmodel'] * experiment['layers']
            alpha = experiment['regul']
            model = RidgeRegression(d_input=d_input, d_output=d_output, alpha=alpha,
                                    to_vec=to_vec)
            X, y = develop_dataset.to_fit_offline_readout()

            logging.info('Fitting Ridge readout.')
            _ = model(X, y)

            X, y = test_dataset.to_evaluate_offline_classifier()
            eval_test = EvaluateOfflineClassifier()
            eval_test.evaluate(y_true=y.numpy(), y_pred=model(X).numpy())
            score = eval_test.accuracy_value

        elif readout == 'mlp':
            device = experiment['device']
            d_input = experiment['dmodel'] * experiment['layers']
            mlplayers = experiment['mlplayers']
            lr = experiment['lr']
            wd = experiment['wd']
            batch = experiment['batch']
            epochs = experiment['epochs']
            patience = experiment['patience']
            plateau = experiment['plateau']

            model = MLP(n_layers=mlplayers, d_input=d_input, d_output=d_output)

            logging.info(f'Moving MLP model to {device}.')
            model.to(device=torch.device(device))

            logging.info('Setting optimizer.')
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=wd)

            develop_dataloader = DataLoader(develop_dataset, batch_size=batch, shuffle=False)

            train_dataset, val_dataset = stratified_split_dataset(dataset=develop_dataset, val_split=val_split)
            train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

            trainer = TrainModel(model=model, optimizer=optimizer, criterion=criterion,
                                 develop_dataloader=develop_dataloader)

            logging.info('Training MLP readout.')
            trainer.early_stopping(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                   patience=patience, num_epochs=epochs, reduce_plateau=plateau,
                                   plot_path=None)

            logging.info(f'Computing reservoir test set.')
            test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

            logging.info('Evaluating model on test set.')
            eval_test = EvaluateClassifier(model=model, num_classes=d_output, dataloader=test_dataloader)
            eval_test.evaluate(saving_path=None)
            score = eval_test.accuracy_value
        else:
            raise ValueError(f'Readout {experiment["readout"]} not recognized.')

        if score > best_score:
            experiment['test_accuracy'] = score
            save_hyperparameters(dictionary=experiment,
                                 file_path=os.path.join('checkpoint', 'results', task, readout, 'best.json'))
            best_score = score


if __name__ == "__main__":
    main()
