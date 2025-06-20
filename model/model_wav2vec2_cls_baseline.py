import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Wav2Vec2Model

class Wav2Vec2ForSpeakerClassification(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        num_classes=10,
        learning_rate=1e-4,
        freeze_feature_extractor=True,
        freeze_encoder_layers=0,
        pooling_method="mean"
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
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # Used to record the val loss / acc of each epoch
        self.val_losses = []
        self.val_accs = []

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling_method == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling_method == "max":
            pooled = hidden_states.max(dim=1).values
        else:
            raise ValueError("Unsupported pooling method")

        logits = self.classifier(pooled)
        loss = self.loss_fn(logits, labels) if labels is not None else None
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

        # Save val_loss and val_acc for epoch end
        self.val_losses.append(outputs["loss"].detach().cpu())
        self.val_accs.append(acc.detach().cpu())
        return {"val_loss": outputs["loss"], "val_acc": acc}

    def on_validation_epoch_end(self):
        # 可以在这里写文件或打印，也可以放在外部 callback 写文件
        mean_loss = torch.stack(self.val_losses).mean().item()
        mean_acc = torch.stack(self.val_accs).mean().item()
        print(f"\nEpoch {self.current_epoch} summary -> val_loss: {mean_loss:.4f}, val_acc: {mean_acc:.4f}")

        # reset
        self.val_losses.clear()
        self.val_accs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        return [optimizer], [scheduler]
