import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model.model_wav2vec2_cls import Wav2Vec2ForSpeakerClassification


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


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    max_len = max([w.shape[0] for w in waveforms])
    padded = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
    return padded, torch.tensor(labels)


def main():
    csv_path = "data/data_fixed.csv"
    batch_size = 2
    learning_rate = 3e-5
    max_epochs = 40
    pretrained_model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str)
    df["meeting_id"] = df["path"].apply(lambda x: "_".join(os.path.basename(x).split("_")[:3]))

    for meeting_id, group in df.groupby("meeting_id"):
        if len(group["label"].unique()) < 2:
            print(f"Skipping {meeting_id}, not enough speakers")
            continue

        print(f"Training for meeting: {meeting_id} with {len(group)} samples and {len(group['label'].unique())} speakers")

        train_df, val_df = train_test_split(group, test_size=0.2, stratify=group["label"], random_state=42)
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([train_df["label"], val_df["label"]], axis=0))

        train_ds = SpeakerDataset(train_df, label_encoder)
        val_ds = SpeakerDataset(val_df, label_encoder)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = Wav2Vec2ForSpeakerClassification(
            pretrained_model_name=pretrained_model,
            num_classes=len(label_encoder.classes_),
            learning_rate=learning_rate,
            freeze_feature_extractor=True,
            freeze_encoder_layers=8,
            pooling_method="mean+std"
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename=f"{meeting_id}-best"
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            num_sanity_val_steps=0  
        )

        trainer.fit(model, train_loader, val_loader)
        print(f"Finished training for meeting {meeting_id}\n\n")


if __name__ == "__main__":
    main()

