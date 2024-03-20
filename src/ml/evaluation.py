import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC


class EvaluateClassifier:
    def __init__(self, model, num_classes, test_dataloader, average='macro'):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.test_dataloader = test_dataloader
        self.accuracy = Accuracy(num_classes=num_classes, average=average).to(device=self.device)
        self.precision = Precision(num_classes=num_classes, average=average).to(device=self.device)
        self.recall = Recall(num_classes=num_classes, average=average).to(device=self.device)
        self.f1 = F1Score(num_classes=num_classes, average=average).to(device=self.device)
        self.roc_auc = AUROC(num_classes=num_classes, average=average, compute_on_step=False).to(device=self.device)
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes).to(device=self.device)

    def __predict(self):
        for input_, label in self.test_dataloader:
            output = self.model(input_)  # outputs of model are logits (raw values)

            # Update metrics
            self.accuracy.update(output, label)
            self.precision.update(output, label)
            self.recall.update(output, label)
            self.f1.update(output, label)
            self.roc_auc.update(output, label)
            self.confusion_matrix.update(output, label)

    def evaluate(self):
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

        print(f"Accuracy: {final_accuracy.item()}")
        print(f"Precision: {final_precision.item()}")
        print(f"Recall: {final_recall.item()}")
        print(f"F1-Score: {final_f1.item()}")
        print(f"ROC-AUC Score: {final_roc_auc.item()}")
        print("Confusion Matrix:\n", final_confusion_matrix)

        # Reset metrics for future use
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.roc_auc.reset()
        self.confusion_matrix.reset()