import torch
import torch.optim as optim

from src.models.s4d.s4d import S4D
from src.task.image.seq2val import Seq2Val
from src.ml.training import max_epochs
from src.utils.temp_data import load_temp_data

# Input and output shape (B, H, L)
torch_input_list_train = load_temp_data('torch_input_list_train')
torch_label_list_train = load_temp_data('torch_label_list_train')
torch_input_list_test = load_temp_data('torch_input_list_test')
torch_label_list_test = load_temp_data('torch_label_list_test')
num_features_input = torch_input_list_train[0].size(1)
s4_model = S4D(d_input=num_features_input, d_state=64)
s4_classifier = Seq2Val(s4_model)
criterion = torch.nn.BCEWithLogitsLoss()  # Classification task: sigmoid layer + BCE loss (more stable)
optimizer = optim.Adam(s4_classifier.parameters(), lr=0.001)
max_epochs(s4_classifier, optimizer, criterion, torch_input_list_train, torch_label_list_train, num_epochs=5)