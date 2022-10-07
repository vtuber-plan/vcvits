import copy
from typing import Optional, Tuple
import random

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

class FeatureExtractor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv0 = nn.Conv1d(1, hidden_size, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(hidden_size, hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm0(self.conv0(x)))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        return x

class FeatureProjection(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_size)
        self.projection = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)
