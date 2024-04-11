import matplotlib.pyplot as plt
import os
import torch
from src.utils.check_device import check_data_device


class TrainModel:
    def __init__(self, model, optimizer, criterion, develop_dataloader):
        self.model = model.to(check_data_device(develop_dataloader))
        self.optimizer = optimizer
        self.criterion = criterion
        self.develop_dataloader = develop_dataloader
        self.training_loss = []
        self.validation_loss = []

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

    def max_epochs(self, num_epochs, checkpoint_path, run_directory=None):
        for epoch in range(num_epochs):
            loss_epoch = self.__epoch(self.develop_dataloader)
            self.training_loss.append(loss_epoch)
            print('[%d] loss: %.3f' % (epoch + 1, loss_epoch))

        torch.save(self.model.state_dict(), checkpoint_path)
        print('Finished Training')

        if run_directory is not None:
            self.__plot(run_directory)

        self.training_loss = []
        self.validation_loss = []

    def early_stopping(self, train_dataloader, val_dataloader, patience, num_epochs=float('inf'),
                       run_directory=None):
        epoch = 0
        buffer = 0
        best_val_loss = float('inf')
        best_model_dict = self.model.state_dict()
        while buffer < patience and epoch < num_epochs:
            train_loss_epoch = self.__epoch(train_dataloader)
            val_loss_epoch = self.__validate(val_dataloader)
            self.training_loss.append(train_loss_epoch)
            self.validation_loss.append(val_loss_epoch)
            if val_loss_epoch < best_val_loss:
                buffer = 0
                best_val_loss = val_loss_epoch
                best_model_dict = self.model.state_dict()
            else:
                buffer += 1
            print('[%d] train_loss: %.3f; val_loss: %.3f' % (epoch + 1, train_loss_epoch, val_loss_epoch))
            epoch += 1

        self.model.load_state_dict(best_model_dict)
        develop_loss = self.__epoch(self.develop_dataloader)

        print('[END] develop_loss: %.3f' % develop_loss)
        print('Finished Training')

        if run_directory is not None:
            torch.save(self.model.state_dict(), os.path.join(run_directory, 'model.pt'))
            self.__plot(run_directory)

        self.training_loss = []
        self.validation_loss = []

    def __plot(self, run_directory):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label='Training loss')
        plt.plot(self.validation_loss, label='Validation loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(run_directory, 'loss.png'))  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory
