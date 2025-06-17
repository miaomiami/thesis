import os
import pandas as pd
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model.model_wav2vec2_cls import Wav2Vec2ForSpeakerClassification


# -----------------------------
# Dataset class
# -----------------------------
class SpeakerDataset(Dataset):
    def __init__(self, df, label_encoder, sample_rate=16000):
        self.df = df
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        label_str = self.df.iloc[idx]["label"]
        label = self.label_encoder.transform([label_str])[0]

        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0), label


# -----------------------------
# Collate function for padding
# -----------------------------
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    padded_waveforms = torch.zeros(len(batch), max_len)

    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :waveform.shape[0]] = waveform

    return padded_waveforms, torch.tensor(labels)


# -----------------------------
# Main training pipeline
# -----------------------------
def main():
    # 🔧 Settings
    csv_path = "data/data_fixed.csv"
    batch_size = 8
    learning_rate = 1e-4
    num_workers = 4
    max_epochs = 10
    pretrained_model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

    # ✅ Load and encode speaker labels
    # ✅ Load data
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str)  # 防止数字被当作 int 处理

    # 🔀 Split data FIRST
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    # ✅ Build label encoder based on train + val
    all_labels = pd.concat([train_df["label"], val_df["label"]], axis=0)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    num_classes = len(label_encoder.classes_)

    print(f"Total samples: {len(df)}, Unique speakers: {num_classes}")

    # 🔀 Split data
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    # 🧾 Datasets and Dataloaders
    train_ds = SpeakerDataset(train_df, label_encoder)
    val_ds = SpeakerDataset(val_df, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)

    # 🧠 Model
    model = Wav2Vec2ForSpeakerClassification(
        pretrained_model_name=pretrained_model,
        num_classes=num_classes,
        learning_rate=learning_rate,
        freeze_feature_extractor=True,
        freeze_encoder_layers=0,
        pooling_method="mean"
    )

    # 📦 Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-speaker-model"
    )

    # 🚀 Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
