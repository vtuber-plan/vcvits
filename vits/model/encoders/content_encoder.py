
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch, torchaudio

from ..hubert.hubert import Hubert
from ... import commons

class ContentEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.hubert = Hubert(out_dim=hidden_channels, extractor_hidden_size=512)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths):
        wav = F.pad(x, ((400 - 320) // 2, (400 - 320) // 2))
        x_encoded, _ = self.hubert.encode(wav)
        x_out = self.hubert.proj(x_encoded).transpose(1, -1)
        x_out = self.proj(x_out)

        x_mask = torch.unsqueeze(commons.sequence_mask((x_lengths / 320).int(), x_out.size(2)), 1).to(x.dtype)
        return x_out, x_mask
