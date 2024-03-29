import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, d_input, d_state):
        super(VanillaRNN, self).__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input
        self.rnn = nn.RNN(input_size=self.d_input, hidden_size=self.d_state, num_layers=1, nonlinearity='tanh',
                          batch_first=False)
        self.fc = nn.Linear(in_features=self.d_state, out_features=self.d_output)
        self.nl = nn.Tanh()

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (L, B, H)
        x = x.permute(2, 0, 1)  # (L, B, H)

        # Forward propagate the RNN
        h, _ = self.rnn(x)  # (L, B, N)
        h = h.reshape(-1, self.d_state)  # (L*B, N)
        y = self.nl(self.fc(h))  # (L*B, H)

        y = y.reshape(-1, self.d_output, x.shape[0])  # (B, H, L)
        return y, None
