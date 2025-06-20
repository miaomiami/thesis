import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Wav2Vec2Model


class Wav2Vec2ForSpeakerClassification(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        num_classes=10,
        learning_rate=3e-5,
        freeze_feature_extractor=True,
        freeze_encoder_layers=8,
        pooling_method="mean+std"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        if freeze_encoder_layers > 0:
            for layer in self.wav2vec2.encoder.layers[:freeze_encoder_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pooling_method = pooling_method
        hidden_size = self.wav2vec2.config.hidden_size
        input_dim = hidden_size * 2 if pooling_method == "mean+std" else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.val_epoch_accs = []

    def _pool_hidden_states(self, hidden_states):
        if self.pooling_method == "mean":
            return hidden_states.mean(dim=1)
        elif self.pooling_method == "max":
            return hidden_states.max(dim=1).values
        elif self.pooling_method == "mean+std":
            mean = hidden_states.mean(dim=1)
            std = hidden_states.std(dim=1)
            return torch.cat([mean, std], dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = self._pool_hidden_states(hidden_states)
        logits = self.classifier(pooled)

        loss = self.loss_fn(logits, labels) if labels is not None else None
        preds = torch.argmax(logits, dim=1)

        return {"loss": loss, "logits": logits, "preds": preds}

    def training_step(self, batch, batch_idx):
        input_values, labels = batch
        outputs = self(input_values, labels=labels)
        self.log("train_loss", outputs["loss"], prog_bar=True)
        return outputs["loss"]

    def on_train_epoch_end(self):
        # Extract the epoch train_loss mean from callback_metrics
        loss = self.trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_epoch_losses.append(loss.item())

    def validation_step(self, batch, batch_idx):
        input_values, labels = batch
        outputs = self(input_values, labels=labels)
        acc = (outputs["preds"] == labels).float().mean()
        self.log("val_loss", outputs["loss"], prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": outputs["loss"], "val_acc": acc}

    def on_validation_epoch_end(self):
        if len(self.train_epoch_losses) == 0:
            # Sanity check stage or not trained, skip
            print(f"Skipping logging on sanity check or before first train epoch")
            return

        val_loss = self.trainer.callback_metrics.get("val_loss").item()
        val_acc = self.trainer.callback_metrics.get("val_acc").item()
        self.val_epoch_losses.append(val_loss)
        self.val_epoch_accs.append(val_acc)
        train_loss = self.train_epoch_losses[-1]
        print(f"Epoch {self.current_epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if self.global_rank == 0:
            with open("results_log.csv", "a") as f:
                if self.current_epoch == 0 and len(self.val_epoch_losses) == 1:
                    f.write("epoch,train_loss,val_loss,val_acc\n")
                f.write(f"{self.current_epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        return [optimizer], [scheduler]

