def check_model_device(model):
    # If model has no parameters, returns the buffer device
    if len(list(model.parameters())) == 0:
        return next(iter(model.buffers())).device
    else:
        return next(iter(model.parameters())).device


def check_data_device(dataloader):
    return next(iter(dataloader))[0].device
