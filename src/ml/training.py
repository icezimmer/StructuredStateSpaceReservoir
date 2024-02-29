def max_epochs(model, optimizer, criterion, torch_input_list_train, torch_label_list_train, num_epochs=10):
    num_batches = len(torch_label_list_train)

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for torch_input, torch_label in zip(torch_input_list_train, torch_label_list_train):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            torch_output = model(torch_input)

            loss = criterion(torch_output, torch_label)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / num_batches))

    print('Finished Training')
