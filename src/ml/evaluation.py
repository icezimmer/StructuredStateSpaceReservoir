import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


def evaluate_classifier(classifier, test_dataloader):
    classifier.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():  # Inference mode, no gradients needed
        for input_, label in test_dataloader:
            input_, label = (input_.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                             label.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

            if len(label.shape) == 1:
                label = label.unsqueeze(1)

            input_ = input_.to(torch.float32)
            label = label.to(torch.float32)

            output = classifier(input_)
            predicted = (output > 0.5).float()  # Convert probabilities to binary predictions
            predictions.extend(predicted.view(-1).numpy())
            true_labels.extend(label.numpy())

    # Convert predictions and labels to appropriate formats if necessary
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f"Accuracy: {accuracy_score(true_labels, predictions)}")
    print(f"Precision: {precision_score(true_labels, predictions)}")
    print(f"Recall: {recall_score(true_labels, predictions)}")
    print(f"F1 Score: {f1_score(true_labels, predictions)}")
    print(f"ROC-AUC Score: {roc_auc_score(true_labels, predictions)}")
