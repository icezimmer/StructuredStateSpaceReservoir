def print_parameters(model, only_trainable=False):
    if only_trainable:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('Parameter name:', name)
                print(param.data)
                print('----------------------------------------------------')
    else:
        for name, param in model.named_parameters():
            print('Parameter name:', name)
            print(param.data)
            print('requires_grad:', param.requires_grad)
            print('----------------------------------------------------')
