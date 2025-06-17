import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
# Train one session
# -----------------------------
def train_single_session(session_id, df_session, pretrained_model, batch_size=8, learning_rate=3e-5, max_epochs=20):
    print(f"\n🔹 Training on session: {session_id}, samples: {len(df_session)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df_session["label"])
    num_classes = len(label_encoder.classes_)

    if num_classes < 2:
        print(f" Skipping session {session_id} — only one speaker.")
        return

    # Split data
    train_df = df_session.sample(frac=0.9, random_state=42)
    val_df = df_session.drop(train_df.index)

    # Dataset & Dataloader
    train_ds = SpeakerDataset(train_df, label_encoder)
    val_ds = SpeakerDataset(val_df, label_encoder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = Wav2Vec2ForSpeakerClassification(
        pretrained_model_name=pretrained_model,
        num_classes=num_classes,
        learning_rate=learning_rate,
        freeze_feature_extractor=True,
        freeze_encoder_layers=8,
        pooling_method="mean+std"
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"{session_id}-best-speaker-model"
    )

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=3,
        verbose=True
    )

    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"✅ Finished training session {session_id}")


# -----------------------------
# Main function
# -----------------------------
def main():
    # Settings
    csv_path = "data/data_fixed.csv"
    pretrained_model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    batch_size = 8
    learning_rate = 3e-5
    max_epochs = 20

    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str)

    # Extract session ID (e.g., 20200706_L_R001S01C01)
    df["session"] = df["path"].apply(lambda x: os.path.basename(x).split("_")[0])

    # Train per session
    for session_id, df_session in df.groupby("session"):
        train_single_session(session_id, df_session, pretrained_model,
                             batch_size=batch_size, learning_rate=learning_rate, max_epochs=max_epochs)

if __name__ == "__main__":
    main()
