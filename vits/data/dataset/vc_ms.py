
import math
import time
import os
import random
from typing import Optional
import numpy as np
import librosa
from librosa import pyin
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torchaudio
 
from vits.mel_processing import MAX_WAV_VALUE, mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from vits.utils import load_filepaths, load_wav_to_torch, load_filepaths_and_text
from ..audio import coarse_f0, estimate_pitch, get_audio, get_pitch, load_audio, shift_audio
import tqdm

import random
import hashlib

def hash_string(s: str) -> str:
    hash_object = hashlib.md5(s.encode("utf-8"))
    return hash_object.hexdigest()

class VoiceConversionMultiSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams, cache_dir: Optional[str]):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value  = hparams.max_wav_value
        self.source_sampling_rate  = hparams.source_sampling_rate
        self.target_sampling_rate  = hparams.target_sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length

        self.resamplers = {}

        random.seed(1234)
        random.shuffle(self.audiopaths)

        self.cache_dir = cache_dir

    def get_item(self, index: int):
        item = self.audiopaths[index]
        audiopath = item[0]
        if len(item) == 1:
            sid = 0
        else:
            sid = int(item[1])
        
        x_audio_path = os.path.join(self.cache_dir, hash_string(f"{audiopath}_{self.source_sampling_rate}") + ".pt")
        if os.path.exists(x_audio_path):
            x_wav = torch.load(x_audio_path)
        else:
            x_audio = load_audio(audiopath, sr=self.source_sampling_rate)
            x_wav = x_audio.unsqueeze(0)

            torch.save(x_wav, x_audio_path)

        x_pitch_mel_path = os.path.join(self.cache_dir, \
            hash_string(
                f"{audiopath}_{self.hparams.filter_length}_{self.hparams.win_length}_{self.hparams.num_pitch}_{self.source_sampling_rate}"
            ) + ".pt")
        if os.path.exists(x_pitch_mel_path):
            x_pitch_mel = torch.load(x_pitch_mel_path)
        else:
            x_pitch_mel = get_pitch(
                audiopath,
                self.hparams.filter_length,
                self.hparams.win_length,
                self.hparams.num_pitch,
                self.source_sampling_rate
            )
            torch.save(x_pitch_mel, x_pitch_mel_path)

        y_audio_path = os.path.join(self.cache_dir, hash_string(f"{audiopath}_{self.target_sampling_rate}") + ".pt")
        if os.path.exists(y_audio_path):
            y_wav = torch.load(y_audio_path)
        else:
            y_audio = load_audio(audiopath, sr=self.target_sampling_rate)
            y_wav = y_audio.unsqueeze(0)

            torch.save(y_wav, y_audio_path)

        return {
            "sid": sid,

            "x_wav": x_wav,
            "x_pitch": x_pitch_mel,

            "y_wav": y_wav,
        }

    def __getitem__(self, index):
        ret = self.get_item(index)
        return ret

    def __len__(self):
        return len(self.audiopaths)
