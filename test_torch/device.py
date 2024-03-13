def test_device(model):
    # Assuming 'model' is your PyTorch model
    for param in model.parameters():
        print(param.device)
