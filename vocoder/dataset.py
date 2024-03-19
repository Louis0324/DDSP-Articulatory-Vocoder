import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import json


class MNGU0Dataset(Dataset):
    def __init__(self, mode, config):
        if mode not in ["train", "val", "test"]:
            raise Exception("mode should only be `train`, `val`, or `test`!")

        json_name = os.path.join(config["data_split_dir"], f"{mode}.json")
        with open(json_name, "r") as f:
            self.file_names = json.load(f)

        self.wav_dir = config["wav_dir"]
        self.ema_dir = config["ema_dir"]
        self.pitch_dir = config["pitch_dir"]
        self.loudness_dir = config["loudness_dir"]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        utt = self.file_names[idx]
        file_name = utt + ".npy"
        wavname = utt + ".wav"

        wav_file, fs = sf.read(os.path.join(self.wav_dir, wavname))
        ema_file = np.load(os.path.join(self.ema_dir, file_name))
        pitch_file = np.load(os.path.join(self.pitch_dir, file_name))
        loudness_file = np.load(os.path.join(self.loudness_dir, file_name))

        wav = torch.from_numpy(wav_file)  # [t, ]
        f0 = torch.from_numpy(pitch_file).unsqueeze(0)  # [1, t]
        loudness = torch.from_numpy(loudness_file).unsqueeze(0)  # [1, t]
        ema = torch.from_numpy(ema_file).transpose(0, 1)  # [12, t]

        return wav, ema, f0, loudness


def train_collate_fn(batch):
    # each segment of ema should be 200-point, i.e. 1s @ 200Hz
    seg_len = 200
    # fs = 16kHz, speech framesize is 80
    framesize = 80

    f0_batch = []
    loudness_batch = []
    ema_batch = []
    wav_batch = []

    for wav, ema, f0, loudness in batch:
        # Randomly crop/pad each file to be 1s long, 200 samples at 200 hz
        # get minimum of ema to define start_idx, since ema length < wav length
        start_idx = np.random.randint(0, max(1, ema.shape[1] - seg_len))
        end_idx = start_idx + seg_len

        f0_batch.append(crop_pad(f0, start_idx, end_idx))
        loudness_batch.append(crop_pad(loudness, start_idx, end_idx))
        ema_batch.append(crop_pad(ema, start_idx, end_idx))
        wav_batch.append(crop_pad(wav.unsqueeze(0), start_idx * framesize, end_idx * framesize).squeeze(0))

    return torch.stack(wav_batch), torch.stack(ema_batch), torch.stack(f0_batch), torch.stack(loudness_batch)


def test_and_validation_collate_fn(batch):
    # fs = 16kHz, speech framesize is 80
    framesize = 80

    f0_batch = []
    loudness_batch = []
    ema_batch = []
    wav_batch = []

    for wav, ema, f0, loudness in batch:
        min_sample_length = ema.shape[-1]
        start_idx = 0
        end_idx = min_sample_length

        f0_batch.append(crop_pad(f0, start_idx, end_idx))
        loudness_batch.append(crop_pad(loudness, start_idx, end_idx))
        ema_batch.append(crop_pad(ema, start_idx, end_idx))
        wav_batch.append(crop_pad(wav.unsqueeze(0), start_idx * framesize, end_idx * framesize).squeeze(0))

    return torch.stack(wav_batch), torch.stack(ema_batch), torch.stack(f0_batch), torch.stack(loudness_batch)


def crop_pad(file, start_idx, end_idx):
    length = file.shape[-1]
    if length >= end_idx:  # if file is large enough crop
        return file[:, start_idx:end_idx]
    else:  # else pad
        pad = torch.zeros(file.shape[0], end_idx - length)
        return torch.concat((file, pad), axis=-1)
