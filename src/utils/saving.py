import pickle
import json
import os
import csv


def save_data(data, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_data(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def save_hyperparameters(args, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        # Convert args namespace to dictionary and save as JSON
        json.dump(vars(args), f, indent=4)


def save_parameters(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for name, param in model.named_parameters():
            file.write(f'Parameter name: {name}\n')
            file.write(f'{param.data.shape}\n')
            file.write(f'requires_grad: {param.requires_grad}\n')
            file.write('----------------------------------------------------\n')
        for name, buffer in model.named_buffers():
            file.write(f'Buffer name: {name}\n')
            file.write(f'{buffer.data.shape}\n')
            file.write(f'requires_grad: {buffer.requires_grad}\n')
            file.write('----------------------------------------------------\n')


def update_results(emissions_path, scores, results_path):
    # Check if the emissions file exists
    if not os.path.exists(emissions_path):
        raise FileNotFoundError(f"The specified emissions file was not found: {emissions_path}")

    # Process the emissions file to retrieve the last row and only take the first 12 columns
    last_row = []
    indices = list(range(0, 5)) + [12]
    with open(emissions_path, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            last_row = [row[i] for i in indices if i < len(row)]

    # If there was no data in the emissions file, stop further processing
    if not last_row:
        raise ValueError("The emissions file does not contain any data.")

    # Add the values of the scores values to the last row
    for key in scores.keys():
        last_row.append(scores[key])

    # Check if results file exists to write header
    file_exists = os.path.exists(results_path)

    # Write the modified data to the results CSV file
    with open(results_path, 'a', newline='') as out_file:
        writer = csv.writer(out_file)
        if not file_exists:
            # Write header only if the file is being created
            writer.writerow(['timestamp', 'project_name', 'run_id',
                             'duration', 'emissions', 'energy_consumed'] + list(scores.keys()))
        writer.writerow(last_row)


def update_hyperparameters(emissions_path, hyperparameters, hyperparameters_path):
    # Check if the emissions file exists
    if not os.path.exists(emissions_path):
        raise FileNotFoundError(f"The specified emissions file was not found: {emissions_path}")

    # Process the emissions file to retrieve the last row and only take the first 12 columns
    last_row = []
    indices = list(range(0, 3))
    with open(emissions_path, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            last_row = [row[i] for i in indices if i < len(row)]

    # If there was no data in the emissions file, stop further processing
    if not last_row:
        raise ValueError("The emissions file does not contain any data.")

    # Add the values of the hyperparameters values to the last row
    for key in hyperparameters.keys():
        last_row.append(hyperparameters[key])

    # Check if results file exists to write header
    file_exists = os.path.exists(hyperparameters_path)

    # Write the modified data to the results CSV file
    with open(hyperparameters_path, 'a', newline='') as out_file:
        writer = csv.writer(out_file)
        if not file_exists:
            # Write header only if the file is being created
            writer.writerow(['timestamp', 'project_name', 'run_id'] + list(hyperparameters.keys()))
        writer.writerow(last_row)
