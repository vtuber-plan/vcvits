import torch
import soundfile as sf
import glob

files = glob.glob("logs/lightning_logs/version_2/checkpoints/*.ckpt")
PATH = sorted(list(files))[-1]

from vits.model.vits import VITS
from vits.mel_processing import spec_to_mel_torch, mel_spectrogram_torch, spectrogram_torch
from vits.utils import load_wav_to_torch, plot_spectrogram_to_numpy
from vits import commons 
from vits.mel_processing import spectrogram_torch
from vits.utils import load_wav_to_torch, load_filepaths_and_text
from vits.text import text_to_sequence, cleaned_text_to_sequence
from vits.text.cleaners import japanese_cleaners
from vits.text.cleaners import chinese_cleaners
from vits.text.cleaners.japanese_mapping import ROMAJI_LIST


model = VITS.load_from_checkpoint(PATH)
hparams = model.hparams

model.eval()

def get_text(text, hparams):
    if getattr(hparams.data, "cleaned_text", False):
        text_norm = cleaned_text_to_sequence(text)
    else:
        text_norm = text_to_sequence(text, hparams.data.text_cleaners)
    if hparams.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

source_text = "早上好呀！"
target_text = chinese_cleaners(text=source_text)
print(target_text)

text_norm = get_text(target_text, model.hparams)

text = text_norm.unsqueeze(0)
text_lengths = torch.LongTensor(1)
text_lengths[0] = text_norm.size(0)

x, x_lengths = text, text_lengths
y_hat, attn, mask, *_ = model.net_g.infer(x, x_lengths, max_len=10000, length_scale=1.2)
y_hat_lengths = mask.sum([1,2]).long() * model.hparams.data.hop_length

sf.write('out.wav', y_hat[0,:,:y_hat_lengths[0]].squeeze(0).detach().numpy(), 48000, subtype='PCM_24')