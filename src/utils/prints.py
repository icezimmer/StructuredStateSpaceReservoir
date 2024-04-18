import json
import csv
import os


def print_parameters(model):
    for name, param in model.named_parameters():
        print('Parameter name:', name)
        print(param.data.shape)
        print('requires_grad:', param.requires_grad)
        print('----------------------------------------------------')


def print_buffers(model):
    for name, buffer in model.named_buffers():
        print('Buffer name:', name)
        print(buffer.data.shape)
        print('requires_grad:', buffer.requires_grad)
        print('----------------------------------------------------')


def save_hyperparameters(args, file_path):
    with open(file_path, 'w') as f:
        # Convert args namespace to dictionary and save as JSON
        json.dump(vars(args), f, indent=4)


def save_parameters(model, file_path):
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

    columns_to_remove = range(13, 31)  # Assuming you want to remove these columns

    # Read and modify data
    modified_data = []
    with open(emissions_path, mode='r', newline='') as in_file:
        reader = csv.reader(in_file)

        # Read the header and modify it
        headers = next(reader)  # This grabs the first row of the CSV, which is the header
        filtered_headers = [header for index, header in enumerate(headers) if index not in columns_to_remove]
        filtered_headers += ['accuracy']
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
