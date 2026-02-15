import torch.nn as nn

class FireLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super(FireLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)

        # Take last layer's hidden state
        out = h_n[-1]

        # Output raw logit (sigmoid applied during inference)
        return self.fc(out)
