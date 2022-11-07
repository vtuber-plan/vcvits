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

        self.pitch_delta = 5
        pitch_shift_list = []
        for i in range(0, self.pitch_delta):
            delta = 2 * i + 1
            for n_step in [-delta, delta]:
                pitch_shift_list.append(
                    T.PitchShift(self.source_sampling_rate, n_step)
                )
        self.pitch_shift_modules = nn.ModuleList(pitch_shift_list)
        for m in self.pitch_shift_modules:
            m.initialize_parameters(torch.tensor([0], dtype=torch.float, device=self.device))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # transform
        if random.random() <= 0.3:
            spec = waveform
        else:
            pitch_shift_index = random.choice(list(range(len(self.pitch_shift_modules))))
            spec = self.pitch_shift_modules[pitch_shift_index](waveform)

        return spec