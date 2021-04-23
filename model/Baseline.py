import torch.nn as nn
import torch


class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, x, x_len):
        x = torch.sum(x,dim=1)
        x_len = x_len.view(-1,1)
        x /= x_len
        return x

        # hidden_sorted = hidden_sorted.view(hidden_sorted.size()[1], -1)
