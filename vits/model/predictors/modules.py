
import torch
from torch import nn
from torch.nn import functional as F


class ConvReLUNorm(torch.nn.Module):
    '''
    https://github.com/NVIDIA/DeepLearningExamples/blob/87aa4b0e855065dbdc64d656a41002048e4be843/PyTorch/SpeechSynthesis/FastPitch/common/layers.py#L76
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)
