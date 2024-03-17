import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


class EvaluateBinaryClassifier:
    def __init__(self, model, test_dataloader, device_name):
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.test_dataloader = test_dataloader
        self.num_batches = len(self.test_dataloader)
        self.accuracy = Accuracy().to(self.device)
        self.precision = Precision().to(self.device)
        self.recall = Recall().to(self.device)
        self.f1 = F1Score().to(self.device)
        #self.roc_auc = AUROC(pos_label=1).to(self.device)

    def __predict(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for input_, label in self.test_dataloader:
            input_, label = (input_.to(self.device, dtype=torch.float32),
                             label.to(self.device, dtype=torch.float32))

            # set feature dimension of label if not present
            if len(label.shape) == 1:
                label = label.unsqueeze(1)

            output = self.model(input_)  # outputs of model are logits (raw values)
            print(torch.sigmoid(output))
            #print(label)
            #output = (torch.sigmoid(output) > 0.5).float()

            tp += torch.sum(((torch.sigmoid(output) > 0.5).int() == 1) & (label == 1)).item()
            fp += torch.sum(((torch.sigmoid(output) > 0.5).int() == 1) & (label == 0)).item()
            tn += torch.sum(((torch.sigmoid(output) > 0.5).int() == 0) & (label == 0)).item()
            fn += torch.sum(((torch.sigmoid(output) > 0.5).int() == 0) & (label == 1)).item()

            # Update metrics
            self.accuracy.update(output, label.int())
            self.precision.update(output, label.int())
            self.recall.update(output, label.int())
            self.f1.update(output, label.int())
            #self.roc_auc.update(output, label.int())

        confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])
        return confusion_matrix

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Inference mode, no gradients needed
            confusion_matrix = self.__predict()

        # Compute final metric values
        final_accuracy = self.accuracy.compute()
        final_precision = self.precision.compute()
        final_recall = self.recall.compute()
        final_f1 = self.f1.compute()
        #final_roc_auc = self.roc_auc.compute()

        print(f"Accuracy: {final_accuracy.item()}")
        print(f"Precision: {final_precision.item()}")
        print(f"Recall: {final_recall.item()}")
        print(f"F1 Score: {final_f1.item()}")
        print("Confusion Matrix:\n", confusion_matrix)
        #print(f"ROC-AUC Score: {final_roc_auc.item()}")

        # Reset metrics for future use
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        #self.roc_auc.reset()


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
                             label.to(self.device, dtype=torch.float32))

            # set feature dimension of label if not present
            if len(label.shape) == 1:
                label = label.unsqueeze(1)

            output = self.model(input_)  # outputs of model are logits (raw values)
            # print(output)
            # print(torch.softmax(output, dim=-1))
            # print(label)
            # print(torch.argmax(label, dim=-1).int())

            # Update metrics
            self.accuracy.update(torch.softmax(output, dim=-1), torch.argmax(label, dim=-1).int())
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

