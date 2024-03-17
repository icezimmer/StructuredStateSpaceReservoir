import torch
import torch.nn as nn


class RNNBlock(nn.Module):
    def __init__(self, d_input, d_state):
        super(RNNBlock, self).__init__()
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


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (L, B, H)
        x = x.permute(2, 0, 1)  # Permuting to match RNN input shape

        # Forward propagate the RNN
        out, _ = self.rnn(x)

        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])

        return out
