import copy
from typing import Optional, Tuple
import random
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from .group_norm import Fp32GroupNorm
from .transpose_last import TransposeLast
from .layer_norm import Fp32LayerNorm

class FeatureExtractor(nn.Module):
    def __init__(self, hidden_size: int, conv_bias: bool = False, dropout: float = 0, mode: str = "default"):
        super().__init__()
        self.dropout = dropout
        self.conv_feature_layers = [(512,9,4)] + [(512,3,2)] * 4 + [(512,2,2)] * 3
        # origin: [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
        self.downsample_num = np.prod([s for d, k, s in self.conv_feature_layers])
        self.conv_layers = nn.ModuleList()

        in_d = 1
        for i, (dim, kernel, stride) in enumerate(self.conv_feature_layers):
            conv = FeatureExtractor.block(in_d, dim, kernel, stride,
                is_layer_norm=mode == "layer_norm",
                is_group_norm=mode == "default" and i == 0,
                conv_bias=conv_bias,
            )
            self.conv_layers.append(conv)
            in_d = dim
    
    @staticmethod
    def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False, dropout=0):
        def make_conv():
            conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            return conv

        assert (
            is_layer_norm and is_group_norm
        ) == False, "layer norm and group norm are exclusive"

        if is_layer_norm:
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(),
                    Fp32LayerNorm(n_out, elementwise_affine=True),
                    TransposeLast(),
                ),
                nn.GELU(),
            )
        elif is_group_norm:
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                Fp32GroupNorm(n_out, n_out, affine=True),
                nn.GELU(),
            )
        else:
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bx1xT -> BxCxT
        for conv in self.conv_layers:
            x = conv(x)

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
