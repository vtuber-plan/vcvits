
from pyexpat import features
import torch

class PreloadAnyVoiceConversionMultiSpeakerCollate():
    def __init__(self, return_ids:bool = False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["x_spec"].size(1) for x in batch]),
            dim=0, descending=True)
        
        hubert_feature_size = batch[0]["x_hubert_features"].shape[0]

        max_x_spec_len = max([x["x_spec"].size(1) for x in batch])
        max_x_wav_len = max([x["x_wav"].size(1) for x in batch])
        max_x_mel_len = max([x["x_mel"].size(1) for x in batch])
        max_x_pitch_len = max([x["x_pitch"].size(1) for x in batch])
        max_x_hubert_features_len = max([x["x_hubert_features"].size(1) for x in batch])

        max_y_spec_len = max([x["y_spec"].size(1) for x in batch])
        max_y_wav_len = max([x["y_wav"].size(1) for x in batch])
        max_y_mel_len = max([x["y_mel"].size(1) for x in batch])
        max_y_pitch_len = max([x["y_pitch"].size(1) for x in batch])
        max_y_hubert_features_len = max([x["y_hubert_features"].size(1) for x in batch])

        sid = torch.LongTensor(len(batch))

        x_spec_lengths = torch.LongTensor(len(batch))
        x_wav_lengths = torch.LongTensor(len(batch))
        x_mel_lengths = torch.LongTensor(len(batch))
        x_pitch_lengths = torch.LongTensor(len(batch))
        x_hubert_features_lengths = torch.LongTensor(len(batch))

        y_spec_lengths = torch.LongTensor(len(batch))
        y_wav_lengths = torch.LongTensor(len(batch))
        y_mel_lengths = torch.LongTensor(len(batch))
        y_pitch_lengths = torch.LongTensor(len(batch))
        y_hubert_features_lengths = torch.LongTensor(len(batch))

        x_spec_padded = torch.zeros(len(batch), batch[0]["x_spec"].size(0), max_x_spec_len, dtype=torch.float32)
        x_wav_padded = torch.zeros(len(batch), 1, max_x_wav_len, dtype=torch.float32)
        x_mel_padded = torch.zeros(len(batch), batch[0]["x_mel"].size(0), max_x_mel_len, dtype=torch.float32)
        x_pitch_padded = torch.zeros(len(batch), max_x_pitch_len, dtype=torch.long)
        x_hubert_features_padded = torch.zeros(len(batch), hubert_feature_size, max_x_hubert_features_len, dtype=torch.float32)

        y_spec_padded = torch.zeros(len(batch), batch[0]["x_spec"].size(0), max_y_spec_len, dtype=torch.float32)
        y_wav_padded = torch.zeros(len(batch), 1, max_y_wav_len, dtype=torch.float32)
        y_mel_padded = torch.zeros(len(batch), batch[0]["x_mel"].size(0), max_y_mel_len, dtype=torch.float32)
        y_pitch_padded = torch.zeros(len(batch), max_y_pitch_len, dtype=torch.long)
        y_hubert_features_padded = torch.zeros(len(batch), hubert_feature_size, max_y_hubert_features_len, dtype=torch.float32)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            sid[i] = row["sid"]

            spec = row["x_spec"]
            x_spec_padded[i, :, :spec.size(1)] = spec
            x_spec_lengths[i] = spec.size(1)

            wav = row["x_wav"]
            x_wav_padded[i, :, :wav.size(1)] = wav
            x_wav_lengths[i] = wav.size(1)

            mel = row["x_mel"]
            x_mel_padded[i, :, :mel.size(1)] = mel
            x_mel_lengths[i] = mel.size(1)

            pitch = row["x_pitch"]
            x_pitch_padded[i, :pitch.size(1)] = pitch
            x_pitch_lengths[i] = pitch.size(1)

            features = row["x_hubert_features"]
            x_hubert_features_padded[i, :, :features.size(1)] = features
            x_hubert_features_lengths[i] = features.size(1)

            spec = row["y_spec"]
            y_spec_padded[i, :, :spec.size(1)] = spec
            y_spec_lengths[i] = spec.size(1)

            wav = row["y_wav"]
            y_wav_padded[i, :, :wav.size(1)] = wav
            y_wav_lengths[i] = wav.size(1)

            mel = row["y_mel"]
            y_mel_padded[i, :, :mel.size(1)] = mel
            y_mel_lengths[i] = mel.size(1)

            pitch = row["y_pitch"]
            y_pitch_padded[i, :pitch.size(1)] = pitch
            y_pitch_lengths[i] = pitch.size(1)

            features = row["y_hubert_features"]
            y_hubert_features_padded[i, :, :features.size(1)] = features
            y_hubert_features_lengths[i] = features.size(1)


        ret = {
            "sid": sid,
            
            "x_spec_values": x_spec_padded,
            "x_spec_lengths": x_spec_lengths,
            "x_wav_values": x_wav_padded,
            "x_wav_lengths": x_wav_lengths,
            "x_mel_values": x_mel_padded,
            "x_mel_lengths": x_mel_lengths,
            "x_pitch_values": x_pitch_padded,
            "x_pitch_lengths": x_pitch_lengths,
            "x_hubert_features_values": x_hubert_features_padded,
            "x_hubert_features_lengths": x_hubert_features_lengths,

            "y_spec_values": y_spec_padded,
            "y_spec_lengths": y_spec_lengths,
            "y_wav_values": y_wav_padded,
            "y_wav_lengths": y_wav_lengths,
            "y_mel_values": y_mel_padded,
            "y_mel_lengths": y_mel_lengths,
            "y_pitch_values": y_pitch_padded,
            "y_pitch_lengths": y_pitch_lengths,
            "y_hubert_features_values": y_hubert_features_padded,
            "y_hubert_features_lengths": y_hubert_features_lengths,

        }

        if self.return_ids:
            ret.update("ids", "ids_sorted_decreasing")
        return ret


class VoiceConversionMultiSpeakerCollate():
    def __init__(self, return_ids:bool = False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["x_wav"].size(1) for x in batch]),
            dim=0, descending=True)

        max_x_wav_len = max([x["x_wav"].size(1) for x in batch])
        max_x_pitch_len = max([x["x_pitch"].size(1) for x in batch])

        max_y_wav_len = max([x["y_wav"].size(1) for x in batch])

        sid = torch.LongTensor(len(batch))

        x_wav_lengths = torch.LongTensor(len(batch))
        x_pitch_lengths = torch.LongTensor(len(batch))

        y_wav_lengths = torch.LongTensor(len(batch))

        x_wav_padded = torch.zeros(len(batch), 1, max_x_wav_len, dtype=torch.float32)
        x_pitch_padded = torch.zeros(len(batch), max_x_pitch_len, dtype=torch.long)

        y_wav_padded = torch.zeros(len(batch), 1, max_y_wav_len, dtype=torch.float32)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            sid[i] = row["sid"]

            wav = row["x_wav"]
            x_wav_padded[i, :, :wav.size(1)] = wav
            x_wav_lengths[i] = wav.size(1)

            pitch = row["x_pitch"]
            x_pitch_padded[i, :pitch.size(1)] = pitch
            x_pitch_lengths[i] = pitch.size(1)

            wav = row["y_wav"]
            y_wav_padded[i, :, :wav.size(1)] = wav
            y_wav_lengths[i] = wav.size(1)

        ret = {
            "sid": sid,
            
            "x_wav_values": x_wav_padded,
            "x_wav_lengths": x_wav_lengths,
            "x_pitch_values": x_pitch_padded,
            "x_pitch_lengths": x_pitch_lengths,

            "y_wav_values": y_wav_padded,
            "y_wav_lengths": y_wav_lengths,
        }

        if self.return_ids:
            ret.update("ids", "ids_sorted_decreasing")
        return ret

