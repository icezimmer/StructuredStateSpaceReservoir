import torch
import torch.optim as optim

from src.models.s4d.s4d import S4D
from src.models.nn.stacked import NaiveStacked
from src.task.image.seq2val import Seq2Val
from test_torch.device import test_device
from src.ml.training import max_epochs
from src.utils.temp_data import load_temp_data

# Input and output shape (B, H, L)
torch_input_list_train = load_temp_data('torch_input_list_train')
torch_label_list_train = load_temp_data('torch_label_list_train')
torch_input_list_test = load_temp_data('torch_input_list_test')
torch_label_list_test = load_temp_data('torch_label_list_test')
num_features_input = torch_input_list_train[0].size(1)

s4d_block = S4D(d_input=num_features_input, d_state=3)
s4d_model = NaiveStacked(block=s4d_block, n_layers=3)
s4d_classifier = Seq2Val(s4d_model)

# Check the device of the first parameter
#s4d_classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
test_device(s4d_classifier)

criterion = torch.nn.BCEWithLogitsLoss()  # Classification task: sigmoid layer + BCE loss (more stable)
optimizer = optim.Adam(s4d_classifier.parameters(), lr=0.001)
max_epochs(s4d_classifier, optimizer, criterion, torch_input_list_train, torch_label_list_train, num_epochs=5)
