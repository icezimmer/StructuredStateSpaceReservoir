import os.path
import matplotlib.pyplot as plt
import json
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
from src.utils.check_device import check_data_device, check_model_device


class EvaluateClassifier:
    def __init__(self, model, num_classes, dataloader, average='macro'):
        self.model = model
        self.device = check_model_device(model=self.model)
        self.dataloader = dataloader
        self.accuracy = Accuracy(num_classes=num_classes, average=average).to(device=self.device)
        self.precision = Precision(num_classes=num_classes, average=average).to(device=self.device)
        self.recall = Recall(num_classes=num_classes, average=average).to(device=self.device)
        self.f1 = F1Score(num_classes=num_classes, average=average).to(device=self.device)
        self.roc_auc = AUROC(num_classes=num_classes, average=average, compute_on_step=False).to(device=self.device)
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes).to(device=self.device)
        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

    def __predict(self):
        for input_, label in self.dataloader:
            input_ = input_.to(device=self.device)
            output = self.model(input_)  # outputs of model are logits (raw values)

            # Update metrics
            self.accuracy.update(output, label)
            self.precision.update(output, label)
            self.recall.update(output, label)
            self.f1.update(output, label)
            self.roc_auc.update(output, label)
            self.confusion_matrix.update(output, label)

    def evaluate(self, run_directory=None, dataset_name=''):
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Inference mode, no gradients needed
            self.__predict()

        # Compute final metric values
        final_accuracy = self.accuracy.compute()
        final_precision = self.precision.compute()
        final_recall = self.recall.compute()
        final_f1 = self.f1.compute()
        final_roc_auc = self.roc_auc.compute()
        final_confusion_matrix = self.confusion_matrix.compute()

        self.accuracy_value = final_accuracy.item()
        self.precision_value = final_precision.item()
        self.recall_value = final_recall.item()
        self.f1_value = final_f1.item()
        self.roc_auc_value = final_roc_auc.item()
        self.confusion_matrix_value = final_confusion_matrix

        print(f"Accuracy: {self.accuracy_value}")
        print(f"Precision: {self.precision_value}")
        print(f"Recall: {self.recall_value}")
        print(f"F1-Score: {self.f1_value}")
        print(f"ROC-AUC Score: {self.roc_auc_value}")
        print("Confusion Matrix:\n", self.confusion_matrix_value)

        if run_directory is not None:
            self.__plot(run_directory, dataset_name)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.roc_auc.reset()
        self.confusion_matrix.reset()

        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

    def __plot(self, run_directory, dataset_name):
        metrics_filename = f'metrics_{dataset_name}.json'
        confusion_matrix_filename = f'confusion_matrix_{dataset_name}.png'

        metrics = {
            "Accuracy": self.accuracy_value,
            "Precision": self.precision_value,
            "Recall": self.recall_value,
            "F1-Score": self.f1_value,
            "ROC-AUC": self.roc_auc_value
        }

        with open(os.path.join(run_directory, metrics_filename), 'w') as f:
            # Convert args namespace to dictionary and save as JSON
            json.dump(metrics, f, indent=4)

        # Plotting the confusion matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(self.confusion_matrix_value.cpu().numpy(), cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(run_directory, confusion_matrix_filename))
        plt.close()
