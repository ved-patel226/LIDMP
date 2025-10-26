import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchmetrics.classification import Accuracy


class CatDogModel(pl.LightningModule):
    def __init__(self) -> None:
        super(CatDogModel, self).__init__()

        # self.model = models.mobilenet_v3_small(pretrained=False)
        self.model = models.mobilenet_v3_large(
            pretrained=False, input_shape=(3, 448, 448)
        )
        self.model.classifier[2] = nn.Linear(
            in_features=1280,
            out_features=64,
        )
        self.model.classifier[3] = nn.Linear(
            in_features=64,  # 1024 for small, 1280 for large
            out_features=2,
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main() -> None:
    from torchsummary import summary

    model = CatDogModel().to("cuda" if torch.cuda.is_available() else "cpu")
    summary(model, (3, 448, 448))


if __name__ == "__main__":
    main()
