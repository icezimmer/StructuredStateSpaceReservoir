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
        print('----------------------------------------------------')


def print_num_trainable_params(model):
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters:', num_trainable_params)
