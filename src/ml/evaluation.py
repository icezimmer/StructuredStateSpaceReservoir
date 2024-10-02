import os.path
import matplotlib.pyplot as plt
import json
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
from tqdm import tqdm
from src.utils.check_device import check_model_device
from sklearn.metrics import accuracy_score, confusion_matrix


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

    def _predict(self):
        for item in tqdm(self.dataloader):
            if len(item) == 3:
                input_, label, length = item
                length = length.to(device=self.device)
            else:
                input_, label = item
                length = None
            input_ = input_.to(device=self.device)
            label = label.to(device=self.device)

            output = self.model(input_, length)  # outputs of model are logits (raw values)

            # Update metrics
            self.accuracy.update(output, label)
            self.precision.update(output, label)
            self.recall.update(output, label)
            self.f1.update(output, label)
            self.roc_auc.update(output, label)
            self.confusion_matrix.update(output, label)

    def evaluate(self, saving_path=None):
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Inference mode, no gradients needed
            self._predict()

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

        if saving_path is not None:
            self._plot(saving_path)

    def reset(self):
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

    def _plot(self, saving_path):
        metrics_path = os.path.join(saving_path, 'metrics.json')
        confusion_matrix_path = os.path.join(saving_path, 'confusion_matrix.png')

        metrics = {
            "Accuracy": self.accuracy_value,
            "Precision": self.precision_value,
            "Recall": self.recall_value,
            "F1-Score": self.f1_value,
            "ROC-AUC": self.roc_auc_value
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            # Convert args namespace to dictionary and save as JSON
            json.dump(metrics, f, indent=4)

        # Plotting the confusion matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(self.confusion_matrix_value.cpu().numpy(), cmap=plt.cm.Purples)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(confusion_matrix_path)
        plt.close()


class EvaluateOfflineClassifier:
    def __init__(self):
        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

    # TODO: compute the rest of metrics
    def evaluate(self, y_true, y_pred, saving_path=None):
        self.accuracy_value = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.confusion_matrix_value = confusion_matrix(y_true=y_true, y_pred=y_pred)

        print(f"Accuracy: {self.accuracy_value}")
        print("Confusion Matrix:\n", self.confusion_matrix_value)

        if saving_path is not None:
            self._plot(saving_path)

    def _plot(self, saving_path):
        metrics_path = os.path.join(saving_path, 'metrics.json')
        confusion_matrix_path = os.path.join(saving_path, 'confusion_matrix.png')

        metrics = {
            "Accuracy": self.accuracy_value,
            "Precision": self.precision_value,
            "Recall": self.recall_value,
            "F1-Score": self.f1_value,
            "ROC-AUC": self.roc_auc_value
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            # Convert args namespace to dictionary and save as JSON
            json.dump(metrics, f, indent=4)

        # Plotting the confusion matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(self.confusion_matrix_value, cmap=plt.cm.Purples)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(confusion_matrix_path)
        plt.close()
