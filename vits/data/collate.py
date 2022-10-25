
from pyexpat import features
import torch

class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids:bool = False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["spec"].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x["text"]) for x in batch])
        max_spec_len = max([x["spec"].size(1) for x in batch])
        max_wav_len = max([x["wav"].size(1) for x in batch])
        max_mel_len = max([x["melspec"].size(1) for x in batch])
        max_pitch_len = max([x["pitch"].size(1) for x in batch])
        max_energy_len = max([x["energy"].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        pitch_lengths = torch.LongTensor(len(batch))
        energy_lengths = torch.LongTensor(len(batch))

        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        spec_padded = torch.zeros(len(batch), batch[0]["spec"].size(0), max_spec_len, dtype=torch.float32)
        wav_padded = torch.zeros(len(batch), 1, max_wav_len, dtype=torch.float32)
        mel_padded = torch.zeros(len(batch), batch[0]["melspec"].size(0), max_mel_len, dtype=torch.float32)
        pitch_padded = torch.zeros(len(batch), 1, max_pitch_len, dtype=torch.float32)
        energy_padded = torch.zeros(len(batch), 1, max_energy_len, dtype=torch.float32)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row["text"]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row["spec"]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row["wav"]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            mel = row["melspec"]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            pitch = row["pitch"]
            pitch_padded[i, :, :pitch.size(1)] = pitch
            pitch_lengths[i] = pitch.size(1)

            energy = row["energy"]
            energy_padded[i, :, :energy.size(1)] = energy
            energy_lengths[i] = energy.size(1)
        
        ret = {
            "text_ids": text_padded, 
            "text_lengths": text_lengths,
            "spec_values": spec_padded,
            "spec_lengths": spec_lengths,
            "wav_values": wav_padded,
            "wav_lengths": wav_lengths,
            "mel_values": mel_padded,
            "mel_lengths": mel_lengths,
            "pitch_values": pitch_padded,
            "pitch_lengths": pitch_lengths,
            "energy_values": energy_padded,
            "energy_lengths": energy_lengths,
        }

        if self.return_ids:
            ret.update("ids", "ids_sorted_decreasing")
        return ret

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class AnyVoiceConversionCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids:bool = False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["x_spec"].size(1) for x in batch]),
            dim=0, descending=True)

        max_x_spec_len = max([x["x_spec"].size(1) for x in batch])
        max_x_wav_len = max([x["x_wav"].size(1) for x in batch])
        max_x_mel_len = max([x["x_mel"].size(1) for x in batch])
        max_x_pitch_len = max([x["x_pitch"].size(1) for x in batch])

        max_y_spec_len = max([x["y_spec"].size(1) for x in batch])
        max_y_wav_len = max([x["y_wav"].size(1) for x in batch])
        max_y_mel_len = max([x["y_mel"].size(1) for x in batch])
        max_y_pitch_len = max([x["y_pitch"].size(1) for x in batch])

        x_spec_lengths = torch.LongTensor(len(batch))
        x_wav_lengths = torch.LongTensor(len(batch))
        x_mel_lengths = torch.LongTensor(len(batch))
        x_pitch_lengths = torch.LongTensor(len(batch))

        y_spec_lengths = torch.LongTensor(len(batch))
        y_wav_lengths = torch.LongTensor(len(batch))
        y_mel_lengths = torch.LongTensor(len(batch))
        y_pitch_lengths = torch.LongTensor(len(batch))

        x_spec_padded = torch.zeros(len(batch), batch[0]["x_spec"].size(0), max_x_spec_len, dtype=torch.float32)
        x_wav_padded = torch.zeros(len(batch), 1, max_x_wav_len, dtype=torch.float32)
        x_mel_padded = torch.zeros(len(batch), batch[0]["x_mel"].size(0), max_x_mel_len, dtype=torch.float32)
        x_pitch_padded = torch.zeros(len(batch), max_x_pitch_len, dtype=torch.long)

        y_spec_padded = torch.zeros(len(batch), batch[0]["x_spec"].size(0), max_y_spec_len, dtype=torch.float32)
        y_wav_padded = torch.zeros(len(batch), 1, max_y_wav_len, dtype=torch.float32)
        y_mel_padded = torch.zeros(len(batch), batch[0]["x_mel"].size(0), max_y_mel_len, dtype=torch.float32)
        y_pitch_padded = torch.zeros(len(batch), max_y_pitch_len, dtype=torch.long)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

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


        ret = {
            "x_spec_values": x_spec_padded,
            "x_spec_lengths": x_spec_lengths,
            "x_wav_values": x_wav_padded,
            "x_wav_lengths": x_wav_lengths,
            "x_mel_values": x_mel_padded,
            "x_mel_lengths": x_mel_lengths,
            "x_pitch_values": x_pitch_padded,
            "x_pitch_lengths": x_pitch_lengths,

            "y_spec_values": y_spec_padded,
            "y_spec_lengths": y_spec_lengths,
            "y_wav_values": y_wav_padded,
            "y_wav_lengths": y_wav_lengths,
            "y_mel_values": y_mel_padded,
            "y_mel_lengths": y_mel_lengths,
            "y_pitch_values": y_pitch_padded,
            "y_pitch_lengths": y_pitch_lengths,
        }

        if self.return_ids:
            ret.update("ids", "ids_sorted_decreasing")
        return ret


class PreloadAnyVoiceConversionCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids:bool = False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
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

