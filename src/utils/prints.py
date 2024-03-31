def print_parameters(model):
    for name, param in model.named_parameters():
        print('Parameter name:', name)
        print(param.data)
        print('requires_grad:', param.requires_grad)
        print('--------------------------------------------')
