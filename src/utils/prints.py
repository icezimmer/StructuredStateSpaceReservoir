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
