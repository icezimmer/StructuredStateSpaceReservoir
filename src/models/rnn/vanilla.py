import torch.nn as nn


class VanillaRecurrent(nn.Module):
    def __init__(self, d_model,
                 dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.rnn = None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        raise NotImplementedError


class VanillaRNN(VanillaRecurrent):
    def __init__(self, d_model):
        super().__init__(d_model)
        self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, nonlinearity='tanh',
                          batch_first=True)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (B, L, H)
        x = x.transpose(-1, -2)  # (B, L, H)

        # Forward propagate the RNN
        y, _ = self.rnn(x)  # (B, L, H)

        y = self.drop(y)

        y = y.transpose(-1, -2)  # (B, H, L)

        return y, None


class VanillaGRU(VanillaRecurrent):
    def __init__(self, d_model,
                 dropout=0.0):
        super().__init__(d_model, dropout)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (B, L, H)
        x = x.transpose(-1, -2)  # (B, L, H)

        # Forward propagate the RNN
        y, _ = self.rnn(x)  # (B, L, H)

        y = self.drop(y)

        y = y.transpose(-1, -2)  # (B, H, L)

        return y, None


class VanillaLSTM(VanillaRecurrent):
    def __init__(self, d_model,
                 dropout=0.0):
        super().__init__(d_model, dropout)
        self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)

    def forward(self, x):
        # x shape: (B, H, L) -> Need to permute it to (B, L, H)
        x = x.transpose(-1, -2)  # (B, L, H)

        # Forward propagate the RNN
        y, _ = self.rnn(x)  # (B, L, H)

        y = self.drop(y)

        y = y.transpose(-1, -2)  # (B, H, L)

        return y, None
