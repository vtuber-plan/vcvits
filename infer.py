import json
import torch
from torch import nn, optim
from torch.nn import functional as F

import soundfile as sf
import glob
import os
import fairseq
from vits.data.audio import coarse_f0, estimate_pitch
from vits.hparams import HParams

files = glob.glob("logs/lightning_logs/version_*/checkpoints/*.ckpt")
PATH = sorted(list(files))[-1]
print(f"Loading....{PATH}")

from vits.model.vcvits import VCVITS
from vits.mel_processing import spec_to_mel_torch, mel_spectrogram_torch, spectrogram_torch
from vits.utils import load_wav_to_torch, plot_spectrogram_to_numpy
from vits import commons 
from vits.mel_processing import spectrogram_torch
from vits.utils import load_wav_to_torch, load_filepaths_and_text
import torchaudio

if torch.cuda.is_available():
    device = "cuda:6"
else:
    device = "cpu"

model = VCVITS.load_from_checkpoint(PATH)
model.eval()
model = model.to(device)

hparams = model.hparams

def get_audio(hparams, filename: str, sr = None, pitch_shift: int = 0):
    audio, sampling_rate = load_wav_to_torch(filename)

    if sr is not None and sampling_rate != sr:
        # not match, then resample
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
        audio = resampler(audio)
        sampling_rate = sr
    
    if pitch_shift != 0:
        shifted_audio = torchaudio.functional.pitch_shift(audio, sampling_rate, pitch_shift)
    else:
        shifted_audio = audio

    shifted_audio_norm = shifted_audio.unsqueeze(0)
    audio_norm = audio.unsqueeze(0)

    # load pitch
    pitch_mel = estimate_pitch(
        audio=shifted_audio.numpy(), sr=sampling_rate, n_fft=hparams.filter_length,
        win_length=hparams.win_length, hop_length=320, method='pyin',
        normalize_mean=None, normalize_std=None, n_formants=1)
    
    coarse_pitch = coarse_f0(pitch_mel)
    pitch_mel = coarse_pitch

    return audio_norm, pitch_mel

def convert(source_audio: str, target_audio: str, speaker_id: int, pitch_shift: int):
    with open("configs/base.json", "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)

    x_wav, x_pitch = get_audio(hparams.data, source_audio, sr=16000, pitch_shift=pitch_shift)

    x_wav = x_wav.unsqueeze(0)
    x_wav_lengths = torch.tensor([x_wav.shape[2]], dtype=torch.long)

    x_pitch = x_pitch.long()
    x_pitch_lengths = torch.tensor([x_pitch.shape[1]], dtype=torch.long)

    x_wav, x_wav_lengths, x_pitch, x_pitch_lengths = x_wav.to(device), x_wav_lengths.to(device), x_pitch.to(device), x_pitch_lengths.to(device)
    
    len_scale = (hparams.data.target_sampling_rate / hparams.data.hop_length) / hparams.data.source_sampling_rate
    sid = torch.tensor([speaker_id], dtype=torch.long).to(device)
    with torch.inference_mode():
        y_hat, mask, (z, z_p, m_p, logs_p) = model.net_g.infer(
                x_wav, x_wav_lengths, x_pitch, x_pitch_lengths,
                sid=sid, length_scale=len_scale, max_len=2000)
    y_hat_lengths = mask.sum([1,2]).long() * model.hparams.data.hop_length

    y_hat = y_hat.to("cpu")

    sf.write(target_audio, y_hat[0,:,:y_hat_lengths[0]].squeeze(0).detach().numpy(), hparams.data.target_sampling_rate, subtype='PCM_24')

convert("ncwlq_01.wav", 'out.wav', 143, 0)