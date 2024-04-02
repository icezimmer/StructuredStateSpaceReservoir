import torch
import torch.nn as nn


class VanillaRecurrent(nn.Module):
    def __init__(self, d_input, d_state):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_input
        self.rnn = None
        self.fc = nn.Linear(in_features=self.d_state, out_features=self.d_output)
        self.nl = nn.Tanh()

    def forward(self, x):
        raise NotImplementedError


class VanillaRNN(VanillaRecurrent):
    def __init__(self, d_input, d_state):
        super().__init__(d_input, d_state)
        self.rnn = nn.RNN(input_size=self.d_input, hidden_size=self.d_state, num_layers=1, nonlinearity='tanh',
                          batch_first=True)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (B, L, H)
        x = x.permute(0, 2, 1)  # (B, L, H)

        # Forward propagate the RNN
        h, _ = self.rnn(x)  # (B, L, N)
        y = self.nl(self.fc(h))  # (B, L, H)

        y = y.permute(0, 2, 1)  # (B, H, L)

        return y, None


class VanillaGRU(VanillaRecurrent):
    def __init__(self, d_input, d_state):
        super().__init__(d_input, d_state)
        self.rnn = nn.GRU(input_size=self.d_input, hidden_size=self.d_state, num_layers=1, batch_first=True)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (B, L, H)
        x = x.permute(0, 2, 1)  # (B, L, H)

        # Forward propagate the RNN
        h, _ = self.rnn(x)  # (B, L, N)
        y = self.nl(self.fc(h))  # (B, L, H)

        y = y.permute(0, 2, 1)  # (B, H, L)

        return y, None
