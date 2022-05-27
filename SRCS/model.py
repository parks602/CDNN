import torch.nn as nn


class DNNModel(nn.Module):
    def __init__(self, len_x):
        super(DNNModel, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(len_x, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        out = self.hidden(x)
        return out.view(-1)
