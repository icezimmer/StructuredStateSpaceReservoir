import torch

def max_epochs(model, optimizer, criterion, train_dataloader, num_epochs=10):
    num_batches = len(train_dataloader)

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for input_, label in train_dataloader:
            input_, label = (input_.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                             label.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

            if len(label.shape) == 1:
                label = label.unsqueeze(1)

            input_ = input_.to(torch.float32)
            label = label.to(torch.float32)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            output = model(input_)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / num_batches))

    print('Finished Training')
