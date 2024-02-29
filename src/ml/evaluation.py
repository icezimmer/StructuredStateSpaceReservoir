import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


def evaluate_classifier(classifier, torch_input_list_test, torch_label_list_test):
    classifier.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():  # Inference mode, no gradients needed
        for torch_input, torch_label in zip(torch_input_list_test, torch_label_list_test):
            torch_output = classifier(torch_input)
            predicted = (torch_output > 0.5).float()  # Convert probabilities to binary predictions
            predictions.extend(predicted.view(-1).numpy())
            true_labels.extend(torch_label.numpy())

    # Convert predictions and labels to appropriate formats if necessary
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f"Accuracy: {accuracy_score(true_labels, predictions)}")
    print(f"Precision: {precision_score(true_labels, predictions)}")
    print(f"Recall: {recall_score(true_labels, predictions)}")
    print(f"F1 Score: {f1_score(true_labels, predictions)}")
    print(f"ROC-AUC Score: {roc_auc_score(true_labels, predictions)}")
