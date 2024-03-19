import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, d_input, d_state):
        super(VanillaRNN, self).__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input
        self.rnn = nn.RNN(input_size=self.d_input, hidden_size=self.d_state, num_layers=1, batch_first=False)
        self.fc = nn.Linear(in_features=self.d_state, out_features=self.d_output)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (L, B, H)
        x = x.permute(2, 0, 1)  # (L, B, H)
        L, B, H = x.shape

        # Forward propagate the RNN
        # h0 = torch.zeros(1, x.shape[1], self.d_state)  # (num_layers, B, d_state)
        y, _ = self.rnn(x)  # (L, B, N)
        y.reshape(L*B, y.shape[2])  # (L*B, N)
        y = self.fc(y)  # (L*B, H)
        y = y.reshape(B, H, L)  # (B, H, L)

        return y, None
