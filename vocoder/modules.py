import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_channel, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_channel, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """
        #### input ####
        x   : in shape [*, input_channel]

        #### output ####
        out : in shape [*, out_dim]

        """
        out = self.layers(x)
        return out


class ResBlock(nn.Module):
    """
    Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf
    Original code:
        https://github.com/dgaddy/silent_speech/blob/master/transformer.py
    """

    def __init__(self, num_ins, num_outs, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)


class ConvNet(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size=3, num_blocks=2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, d_model, kernel_size, padding=(kernel_size - 1) // 2),
            *[ResBlock(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2) for _ in range(num_blocks - 1)],
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).

        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)
