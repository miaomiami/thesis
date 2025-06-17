import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2ForSpeakerClassification(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name="TencentGameMate/wav2vec2-large-chinese",
        num_classes=10,
        learning_rate=1e-4,
        freeze_feature_extractor=True,
        freeze_encoder_layers=0,
        pooling_method="mean"
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained wav2vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        # Freeze feature extractor (conv layers)
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # Optionally freeze some transformer layers
        if freeze_encoder_layers > 0:
            for layer in self.wav2vec2.encoder.layers[:freeze_encoder_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Pooling strategy: mean or max
        self.pooling_method = pooling_method

        # Classification head
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        # Mean or max pooling over time
        if self.pooling_method == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling_method == "max":
            pooled = hidden_states.max(dim=1).values
        else:
            raise ValueError("Unsupported pooling method")

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx):
        input_values, labels = batch
        outputs = self(input_values, labels=labels)
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        input_values, labels = batch
        outputs = self(input_values, labels=labels)
        preds = torch.argmax(outputs["logits"], dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", outputs["loss"], prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": outputs["loss"], "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        return [optimizer], [scheduler]
