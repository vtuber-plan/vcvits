
import math
import torch
from torch import nn
from torch.nn import functional as F

from .modules import ConvReLUNorm

class EnergyPredictor(nn.Module):
    '''
    参考Nvidia的FastPitch
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/model.py#L90
    '''
    def __init__(self, in_channels, filter_channels, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(EnergyPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(in_channels if i == 0 else filter_channels, filter_channels,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_channels, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out
