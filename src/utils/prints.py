import json


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
