import torch
from src.utils.check_device import check_data_device


class TrainModel:
    def __init__(self, model, optimizer, criterion, develop_dataloader):
        self.model = model.to(check_data_device(develop_dataloader))
        self.optimizer = optimizer
        self.criterion = criterion
        self.develop_dataloader = develop_dataloader

    def __epoch(self, dataloader):
        self.model.train()

        running_loss = 0.0
        for input_, label in dataloader:
            self.optimizer.zero_grad()
            output = self.model(input_)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def __validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for input_, label in dataloader:
                output = self.model(input_)
                running_loss += self.criterion(output, label).item()

            return running_loss / len(dataloader)

    def max_epochs(self, num_epochs, checkpoint_path):
        for epoch in range(num_epochs):
            loss_epoch = self.__epoch(self.develop_dataloader)
            print('[%d] loss: %.3f' % (epoch + 1, loss_epoch))

        torch.save(self.model.state_dict(), checkpoint_path)
        print('Finished Training')

    def early_stopping(self, train_dataloader, val_dataloader, patience, checkpoint_path, num_epochs=float('inf')):
        epoch = 0
        buffer = 0
        best_val_loss = float('inf')
        while buffer < patience and epoch < num_epochs:
            train_loss_epoch = self.__epoch(train_dataloader)
            val_loss_epoch = self.__validate(val_dataloader)
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                buffer = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                buffer += 1
            print('[%d] train_loss: %.3f; val_loss: %.3f' % (epoch + 1, train_loss_epoch, val_loss_epoch))
            epoch += 1

        self.model.load_state_dict(torch.load(checkpoint_path))
        develop_loss = self.__epoch(self.develop_dataloader)
        torch.save(self.model.state_dict(), checkpoint_path)

        print('[END] develop_loss: %.3f' % develop_loss)
        print('Finished Training')
