import logging


def print_parameters(model):
    for name, param in model.named_parameters():
        logging.info(f'Parameter name: {name} = {param.data.shape}, requires_grad = {param.requires_grad}')


def print_buffers(model):
    for name, buffer in model.named_buffers():
        logging.info(f'Buffer name: {name} = {buffer.data.shape}')


def print_num_trainable_params(model):
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_trainable_params}')
