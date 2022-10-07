
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch, torchaudio

class VoiceEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

    def forward(self, x, x_lengths):
        units = self.hubert.units(x)
        return units
