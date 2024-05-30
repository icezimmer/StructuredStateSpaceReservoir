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


def create_results(emissions_path, output_path):
    if os.path.exists(output_path):
        return  # Stop execution if the file exists

    # Define the columns to remove (zero-indexed)
    columns_to_remove = list(range(5, 12)) + list(range(13, 31))

    # Read and modify data
    modified_data = []
    with open(emissions_path, mode='r', newline='') as in_file:
        reader = csv.reader(in_file)

        # Read the header and modify it
        headers = next(reader)  # This grabs the first row of the CSV, which is the header
        filtered_headers = [header for index, header in enumerate(headers) if index not in columns_to_remove]
        filtered_headers.append('accuracy')  # Add 'accuracy' to the header
        modified_data.append(filtered_headers)

        # Read and modify each row
        for row in reader:
            filtered_row = [value for index, value in enumerate(row) if index not in columns_to_remove]
            filtered_row.append('')  # Add an empty column for 'accuracy'
            modified_data.append(filtered_row)

    # Write the modified data to a new CSV file
    with open(output_path, mode='w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(modified_data)


def add_score(metrics_test_path, results_path):
    if not os.path.exists(metrics_test_path):
        print("Metrics file does not exist.")
        return

    if not os.path.exists(results_path):
        print("Results file does not exist.")
        return

    with open(metrics_test_path, 'r') as json_file:
        metrics = json.load(json_file)

    if "Accuracy" not in metrics:
        print("Accuracy data is missing in metrics.")
        return

    accuracy = metrics["Accuracy"]

    with open(results_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        header = rows[0]
        accuracy_index = header.index('accuracy') if 'accuracy' in header else len(header)
        if 'accuracy' not in header:
            header.append('accuracy')
            for row in rows[1:]:
                row.append('')

        rows[-1][accuracy_index] = accuracy  # Set accuracy in the last row's correct column

        with open(results_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)


def update_results(emissions_path, metrics_test_path, results_path):
    # Check if the metrics file exists
    if not os.path.exists(emissions_path):
        raise FileNotFoundError(f"The specified emissions file was not found: {emissions_path}")

        # Check if the metrics file exists and is accessible
    if not os.path.exists(metrics_test_path):
        raise FileNotFoundError(f"The specified metrics file was not found: {metrics_test_path}")

    # Load accuracy from the metrics file
    with open(metrics_test_path, 'r') as json_file:
        metrics = json.load(json_file)

    if "Accuracy" not in metrics:
        raise KeyError("The metrics file does not contain the 'Accuracy' key.")
    accuracy = metrics["Accuracy"]

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

    # Add the accuracy to the last row
    last_row.append(accuracy)

    # Check if results file exists to write header
    file_exists = os.path.exists(results_path)

    # Write the modified data to the results CSV file
    with open(results_path, 'a', newline='') as out_file:
        writer = csv.writer(out_file)
        if not file_exists:
            # Write header only if the file is being created
            writer.writerow(['timestamp', 'project_name', 'run_id',
                             'duration', 'emissions', 'energy_consumed', 'accuracy'])
        writer.writerow(last_row)
