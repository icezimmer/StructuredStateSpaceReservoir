def check_model_device(model):
    return next(iter(model.parameters())).device


def check_data_device(dataloader):
    return next(iter(dataloader))[0].device
