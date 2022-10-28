import json
import torch
from torch import nn, optim
from torch.nn import functional as F

import soundfile as sf
import glob
import os
import fairseq
from vits.data.dataset import coarse_f0, estimate_pitch
from vits.hparams import HParams
from vits.model.preload_vcvits import PreloadVCVITS

files = glob.glob("logs/lightning_logs/version_0/checkpoints/*.ckpt")
PATH = sorted(list(files))[-1]

from vits.model.vcvits import VCVITS
from vits.mel_processing import spec_to_mel_torch, mel_spectrogram_torch, spectrogram_torch
from vits.utils import load_wav_to_torch, plot_spectrogram_to_numpy
from vits import commons 
from vits.mel_processing import spectrogram_torch
from vits.utils import load_wav_to_torch, load_filepaths_and_text
import torchaudio

if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = "cpu"

model = PreloadVCVITS.load_from_checkpoint(PATH)
model.eval()
model = model.to(device)

hparams = model.hparams

models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([hparams.data.hubert_ckpt])
hubert = models[0].to(device)
hubert.eval()

resamplers = {}

def get_audio(hparams, filename: str, sr = None):
    audio, sampling_rate = load_wav_to_torch(filename)
    if len(audio.shape) > 1:
        audio = torch.mean(audio, dim=1)
    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sampling_rate in resamplers:
            resampler = resamplers[sampling_rate]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[sampling_rate] = resampler
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    # load spec
    spec = spectrogram_torch(audio_norm, hparams.filter_length, sr, hparams.hop_length, hparams.win_length, center=False)
    spec = torch.squeeze(spec, 0)

    # load mel
    melspec = spec_to_mel_torch(spec, hparams.filter_length, hparams.n_mel_channels, sr, hparams.mel_fmin, hparams.mel_fmax)
    melspec = torch.squeeze(melspec, 0)

    # load pitch
    pitch_mel = estimate_pitch(
        audio=audio.numpy(), sr=sampling_rate, n_fft=hparams.filter_length,
        win_length=hparams.win_length, hop_length=320, method='pyin',
        normalize_mean=None, normalize_std=None, n_formants=1)
    
    coarse_pitch = coarse_f0(pitch_mel)
    pitch_mel = coarse_pitch

    # load features
    wav = F.pad(audio_norm, ((400 - 320) // 2, (400 - 320) // 2))
    wav_input = wav.squeeze(1).to(device)

    hubert_features, _ = hubert.extract_features(wav_input)
    hubert_features = hubert_features.transpose(1, -1).squeeze(0)

    hubert_features = hubert_features.to("cpu")

    return spec, audio_norm, melspec, pitch_mel, hubert_features

def convert(source_audio, target_audio):
    with open("configs/paimoon_base_vc_ms_fast.json", "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)

    x_spec, x_wav, x_melspec, x_pitch, x_features = get_audio(hparams.data, source_audio, sr=16000)

    x_wav = x_wav.unsqueeze(0)
    x_wav_lengths = torch.tensor([x_wav.shape[2]], dtype=torch.long)

    x_features = x_features.unsqueeze(0)
    x_features_lengths = torch.tensor([x_features.shape[2]], dtype=torch.long)

    x_pitch = x_pitch.long()
    x_pitch_lengths = torch.tensor([x_pitch.shape[1]], dtype=torch.long)

    x_features, x_features_lengths, x_pitch, x_pitch_lengths = x_features.to(device), x_features_lengths.to(device), x_pitch.to(device), x_pitch_lengths.to(device)
    
    len_scale = (hparams.data.target_sampling_rate / hparams.data.hop_length) \
                    / (hparams.data.source_sampling_rate / hparams.data.hubert_downsample)
    sid = torch.tensor([0], dtype=torch.long).to(device)
    y_hat, mask, (z, z_p, m_p, logs_p) = model.net_g.infer(
            x_features, x_features_lengths, x_pitch, x_pitch_lengths,
            sid=sid, length_scale=len_scale, max_len=1000)
    y_hat_lengths = mask.sum([1,2]).long() * model.hparams.data.hop_length

    y_hat = y_hat.to("cpu")

    sf.write(target_audio, y_hat[0,:,:y_hat_lengths[0]].squeeze(0).detach().numpy(), 48000, subtype='PCM_24')

convert("dataset/LJSpeech/LJ001-0001.wav", 'out.wav')