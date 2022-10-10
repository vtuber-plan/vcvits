
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch

from vits.model.transformer.relative_attention_transformer import TransformerEncoder

from ..natsubert.natsubert import NatsuBert
from ..hubert.hubert import Hubert
from ... import commons

class NatsuBertContentEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 n_fft=2048,
                 hop_size=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.out_channels = out_channels
        self.hubert = NatsuBert(out_dim=hidden_channels, extractor_hidden_size=512)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        wav = F.pad(x, (self.n_fft // 8, self.n_fft // 8))
        # wav = x
        x_encoded, _ = self.hubert.encode(wav)
        x_out = self.hubert.proj(x_encoded).transpose(1, -1)

        # n_downsample = self.hubert.feature_extractor.downsample_num
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_out.size(2)), 1).to(x.dtype)
   
        stats = self.proj(x_out) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x_out, m, logs, x_mask

class HubertContentEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 n_fft=2048,
                 hop_size=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.out_channels = out_channels
        
        self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.pitch_proj = nn.Conv1d(1, hidden_channels, 1)
        
        self.encoder = TransformerEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, pitch, pitch_lengths):
        wav = F.pad(x, ((400 - 320) // 2, (400 - 320) // 2))
        x_encoded, _ = self.hubert.encode(wav)
        hubert_out = self.hubert.proj(x_encoded).transpose(1, -1)

        hubert_out = hubert_out + self.pitch_proj(pitch)

        # n_downsample = self.hubert.feature_extractor.downsample_num
        x_mask = torch.unsqueeze(commons.sequence_mask((x_lengths/320).int(), hubert_out.size(2)), 1).to(x.dtype)
   
        x_out = self.encoder(hubert_out * x_mask, x_mask)

        stats = self.proj(hubert_out) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x_out, m, logs, x_mask
