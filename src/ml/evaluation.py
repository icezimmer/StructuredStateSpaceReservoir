import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


class EvaluateClassifier:
    def __init__(self, model, num_classes, test_dataloader, device_name):
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.test_dataloader = test_dataloader
        self.accuracy = Accuracy(num_classes=num_classes).to(self.device)
        #self.roc_auc = AUROC(pos_label=1).to(self.device)

    def __predict(self):
        for input_, label in self.test_dataloader:
            input_, label = (input_.to(self.device, dtype=torch.float32),
                             label.to(self.device, dtype=torch.long))

            output = self.model(input_)  # outputs of model are logits (raw values)

            # Update metrics
            self.accuracy.update(output, label)
            #self.roc_auc.update(output, label.int())

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Inference mode, no gradients needed
            self.__predict()

        # Compute final metric values
        final_accuracy = self.accuracy.compute()
        #final_roc_auc = self.roc_auc.compute()

        print(f"Accuracy: {final_accuracy.item()}")

        #print(f"ROC-AUC Score: {final_roc_auc.item()}")

        # Reset metrics for future use
        self.accuracy.reset()
        #self.roc_auc.reset()
