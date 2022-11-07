import torch
import torchaudio
import torchaudio.transforms as T

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

class SpeechConversionAudioPipeline(torch.nn.Module):
    def __init__(
        self,
        source_sampling_rate: int,
        device: str
    ):
        super().__init__()
        self.source_sampling_rate = source_sampling_rate
        self.device = device
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # transform
        if random.random() < 0.3:
            pitch_shift = 0
            waveform_out = waveform
        else:
            pitch_shift = random.randint(-12, 12)
            waveform_out = torchaudio.functional.pitch_shift(waveform, self.source_sampling_rate, pitch_shift)

        return waveform_out