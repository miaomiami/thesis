# dataset.py
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class SpeakerDataset(Dataset):
    def __init__(self, csv_path, label_encoder=None, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.df["label"] = self.df["label"].astype(str)
        self.sample_rate = sample_rate

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.df["label"])
        else:
            self.label_encoder = label_encoder

        self.df["encoded_label"] = self.label_encoder.transform(self.df["label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = row["encoded_label"]

        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform.squeeze(0), label


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    padded_waveforms = torch.zeros(len(batch), max_len)

    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :waveform.shape[0]] = waveform

    return padded_waveforms, torch.tensor(labels)
