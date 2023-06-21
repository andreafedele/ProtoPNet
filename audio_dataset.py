import os
# import numpy as np
import pandas as pd

import torch
import librosa
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, annotations_file,  audio_dir, target_sample_rate, num_samples, transformation, power_or_db):
        self.annotations = pd.read_csv(annotations_file, dtype={'label':'string'})
        # self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # self.device = device
        #self.transformation = transformation.to(self.device)
        self.transformation = transformation
        self.power_or_db = power_or_db

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        # signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        if self.power_or_db == 'd':
            signal = librosa.power_to_db(signal)

        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _get_audio_sample_path(self, index):
        return os.path.join(self.audio_dir, self._get_audio_sample_label(index), self.annotations.iloc[index, 0])

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]
    
