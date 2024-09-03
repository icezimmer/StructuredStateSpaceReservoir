import argparse
import os
from src.utils.saving import load_data
import logging
import random
import itertools
from src.utils.experiments import read_yaml_to_dict
from src.readout.ridge import RidgeRegression
from src.ml.evaluation import EvaluateOfflineClassifier
from sklearn.linear_model import RidgeClassifierCV
from src.utils.saving import save_hyperparameters
import torch


loss = {
    'cross_entropy': torch.nn.CrossEntropyLoss()
}


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run readout based on YAML configuration.")
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
    to_vec = architecture['to_vec']
    d_input = develop_dataset[0][0].shape[-2]  # From (*, nP, L=1)

    experiments = {
        'regul': [0.8, 1.5, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0]
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

    # X, y = develop_dataset.to_fit_offline_readout()
    # logging.info('Ridge Classifier Cross Validation.')
    # clf = RidgeClassifierCV(alphas=experiments['regul'], cv=3).fit(X.numpy(), y.numpy())
    # X, y = test_dataset.to_evaluate_offline_classifier()
    # y_pred = clf.predict(X.numpy())
    # score = clf.score(y_pred.numpy(), y.numpy())
    # print(score)

    # Generate and execute sampled experiments
    best_score = 0.0

    for experiment in sampled_experiments:
        print(experiment)
        X, y = develop_dataset.to_fit_offline_readout()
        model = RidgeRegression(d_input=d_input, d_output=d_output, alpha=experiment['regul'],
                                to_vec=to_vec)

        logging.info('Fitting Ridge readout.')
        _ = model(X, y)

        X, y = test_dataset.to_evaluate_offline_classifier()
        eval_test = EvaluateOfflineClassifier()
        eval_test.evaluate(y_true=y.numpy(), y_pred=model(X).numpy())
        score = eval_test.accuracy_value

        if score > best_score:
            experiment['test_accuracy'] = score
            save_hyperparameters(dictionary=experiment,
                                 file_path=os.path.join('checkpoint', 'results', args.task, 'ridge', 'best.json'))
            best_score = score


if __name__ == "__main__":
    main()
