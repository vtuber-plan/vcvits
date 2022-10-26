import glob
import os
import json
import argparse
import itertools
import math
import torch
import torchaudio
import fairseq
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from vits.model.preload_vcvits import PreloadVCVITS

import vits.utils as utils
from vits.data.collate import PreloadAnyVoiceConversionMultiSpeakerCollate

import pytorch_lightning as pl

from vits.hparams import HParams
from vits.model.vits import VITS
from vits.data.dataset import PreloadAnyVoiceConversionMultiSpeakerLoader

def get_hparams() -> HParams:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/paimoon_base_vc_ms_fast.json", help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def load_hubert(path: str, device: str):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    hubert = models[0].to(device)
    hubert.eval()
    return hubert

def preprocess(hparams, files, sr=16000):
    # load hubert
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    hubert = None

    audiopaths = utils.load_filepaths(files)

    resamplers = {}

    print("Preprocessing dataset...")
    with torch.inference_mode():
        for filename in tqdm.tqdm(audiopaths):
            feature_filename = filename.replace(".wav", f"_{sr}.feature.pt")
            if os.path.exists(feature_filename):
                continue

            audio, sampling_rate = utils.load_wav_to_torch(filename)
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
            audio_norm = audio / hparams.data.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)

            wav = F.pad(audio_norm, ((400 - 320) // 2, (400 - 320) // 2))
            wav_input = wav.squeeze(1).to(device)

            if hubert is None:
                hubert = load_hubert(hparams.data.hubert_ckpt, device)

            hubert_features, _ = hubert.extract_features(wav_input)
            hubert_features = hubert_features.transpose(1, -1).squeeze(0)

            hubert_features = hubert_features.to("cpu")
            torch.save(hubert_features, feature_filename)


def main():
    hparams = get_hparams()
    pl.utilities.seed.seed_everything(hparams.train.seed)

    train_dataset = PreloadAnyVoiceConversionMultiSpeakerLoader(hparams.data.training_files, hparams.data)
    valid_dataset = PreloadAnyVoiceConversionMultiSpeakerLoader(hparams.data.validation_files, hparams.data)

    preprocess(hparams, hparams.data.training_files, hparams.data.source_sampling_rate)
    preprocess(hparams, hparams.data.validation_files, hparams.data.source_sampling_rate)
    
    collate_fn = PreloadAnyVoiceConversionMultiSpeakerCollate()
    train_loader = DataLoader(train_dataset, batch_size=hparams.train.batch_size, num_workers=16, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = PreloadVCVITS(**hparams)

    trainer_params = {
        "accelerator": "gpu",
        "devices": [0],
        # "strategy": "ddp",
    }

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
  main()
