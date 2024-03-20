import torch


class TrainModel:
    def __init__(self, model, optimizer, criterion, train_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.num_batches = len(self.train_dataloader)

    def __epoch(self):
        self.model.train()

        running_loss = 0.0
        for input_, label in self.train_dataloader:

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            output = self.model(input_)

            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / self.num_batches

    def max_epochs(self, num_epochs):
        for epoch in range(num_epochs):  # Loop over the dataset multiple times
            loss_epoch = self.__epoch()
            print('[%d] loss: %.3f' % (epoch + 1, loss_epoch))

        print('Finished Training')
