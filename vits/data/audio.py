
import os
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
from ..utils import load_wav_to_torch

def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch

def estimate_pitch(audio: np.ndarray, sr: int, n_fft: int, win_length: int, hop_length: int,
                    method='pyin', normalize_mean=None, normalize_std=None, n_formants=1):
    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':
        snd, sr = audio, sr
        pad_size = int((n_fft-hop_length)/2)
        snd = np.pad(snd, (pad_size, pad_size), mode='reflect')

        pitch_mel, voiced_flag, voiced_probs = pyin(
            snd,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=win_length,
            hop_length=hop_length,
            center=False,
            pad_mode='reflect')
        # assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        # pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError
    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel

def coarse_f0(f0: torch.FloatTensor, f0_min:float=50, f0_max:float=1100, f0_bin:int=512):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = torch.round(f0_mel)
    assert f0_coarse.max() < f0_bin and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(),)
    return f0_coarse

resamplers = {}

def get_audio_preload(filename: str, 
                max_wav_value: int,
                filter_length: int,
                hop_length: int,
                win_length: int,
                n_mel_channels: int,
                mel_fmin: int,
                mel_fmax: int,
                hubert_channels: int,
                num_pitch: int,
                pitch_shift: int = 0,
                sr: Optional[int] = None, load_features: bool = False):
    global resamplers
    audio, sampling_rate = load_wav_to_torch(filename)

    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sr in resamplers:
            resampler = resamplers[(sampling_rate, sr)]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[(sampling_rate, sr)] = resampler
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    original_audio = audio
    if pitch_shift != 0:
        audio = torchaudio.functional.pitch_shift(audio, sampling_rate, pitch_shift)
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    original_audio_norm = original_audio / max_wav_value
    original_audio_norm = original_audio_norm.unsqueeze(0)

    # load spec
    spec_filename = filename.replace(".wav", f"_{sampling_rate}.spec.pt")
    if os.path.exists(spec_filename):
        spec = torch.load(spec_filename)
    else:
        spec = spectrogram_torch(audio_norm, filter_length, sampling_rate, hop_length, win_length, center=False)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename)
    
    # load mel
    mel_filename = filename.replace(".wav", f"_{sampling_rate}.mel.pt")
    if os.path.exists(mel_filename):
        melspec = torch.load(mel_filename)
    else:
        melspec = spec_to_mel_torch(spec, filter_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        melspec = torch.squeeze(melspec, 0)
        torch.save(melspec, mel_filename)

    # load pitch
    pitch_filename = filename.replace(".wav", f"_{sampling_rate}.pitch.pt")
    if os.path.exists(pitch_filename):
        pitch_mel = torch.load(pitch_filename)
    else:
        pitch_mel = estimate_pitch(
            audio=original_audio_norm.numpy(), sr=sampling_rate, n_fft=filter_length,
            win_length=win_length, hop_length=320, method='pyin',
            normalize_mean=None, normalize_std=None, n_formants=1)
        
        coarse_pitch = coarse_f0(pitch_mel, f0_bin=num_pitch)
        pitch_mel = coarse_pitch
        torch.save(pitch_mel, pitch_filename)
    
    # load features
    feature_filename = filename.replace(".wav", f"_{sampling_rate}.feature.pt")
    if os.path.exists(feature_filename):
        hubert_features = torch.load(feature_filename)
    else:
        if load_features:
            raise Exception("Please preprocess the dataset before training")
        else:
            hubert_features = torch.zeros(hubert_channels, 1)

    return spec, audio_norm, melspec, pitch_mel, hubert_features

def load_audio(filename: str, sr: Optional[int] = None):
    global resamplers
    audio, sampling_rate = load_wav_to_torch(filename)

    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sr in resamplers:
            resampler = resamplers[(sampling_rate, sr)]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[(sampling_rate, sr)] = resampler
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    return audio

def shift_audio(audio: torch.FloatTensor, sr: Optional[int] = None, pitch_shift: int = 0):
    if pitch_shift != 0:
        shifted_audio = torchaudio.functional.pitch_shift(audio, sr, pitch_shift)
    else:
        shifted_audio = audio

    return shifted_audio

def get_audio(audio_norm: torch.FloatTensor, 
        max_wav_value: int,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
        mel_fmin: int,
        mel_fmax: Optional[int],
        hubert_channels: int,
        num_pitch: int,
        sr: Optional[int] = None,
        load_spec: bool=False,
        load_mel: bool=False):

    # load spec
    if load_spec:
        spec = spectrogram_torch(audio_norm, filter_length, sr, hop_length, win_length, center=False)
        spec = torch.squeeze(spec, 0)
    else:
        spec = None

    # load mel
    if load_mel:
        melspec = spec_to_mel_torch(spec, filter_length, n_mel_channels, sr, mel_fmin, mel_fmax)
        melspec = torch.squeeze(melspec, 0)
    else:
        melspec = None

    return spec, melspec


def get_pitch(filename: str,
        filter_length: int,
        win_length: int,
        num_pitch: int,
        sr: Optional[int] = None):
    global resamplers
    audio, sampling_rate = load_wav_to_torch(filename)

    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sr in resamplers:
            resampler = resamplers[(sampling_rate, sr)]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[(sampling_rate, sr)] = resampler
        audio = resampler(audio)
        sampling_rate = sr

    pitch_mel = estimate_pitch(
        audio=audio.numpy(), sr=sampling_rate, n_fft=filter_length,
        win_length=win_length, hop_length=320, method='pyin',
        normalize_mean=None, normalize_std=None, n_formants=1)
    
    coarse_pitch = coarse_f0(pitch_mel, f0_bin=num_pitch)
    pitch_mel = coarse_pitch

    return pitch_mel