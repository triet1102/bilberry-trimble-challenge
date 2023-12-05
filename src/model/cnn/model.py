import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


def save_model(model: torch.nn.Module, file_path: str) -> None:
    """
    save the model
    """
    torch.save(model.state_dict(), file_path)


def get_backbone(
    config: dict,
) -> tuple[torch.nn.Module, transforms.Compose, transforms.Compose]:
    """get the model and transforms

    Args:
        config: the configuration
        file_path: if provided, load the trained model from the file path
                   if not provided, load the model from timm

    Returns:
        the model and train/val transforms
    """
    backbone = timm.create_model(
        model_name=config.model.backbone,
        pretrained=True,
        num_classes=0,
    )

    if config.model.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone


class ClassificationModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.backbone = get_backbone(config=config)
        self.feature_dim = self.backbone.head.num_features
        self.classifier = (
            nn.Sequential(
                nn.Linear(
                    in_features=self.feature_dim, out_features=config.model.hidden_units
                ),
                nn.ReLU(),
                nn.Linear(config.model.hidden_units, config.model.num_classes),
            )
            if config.model.hidden_units > 0
            else nn.Linear(self.feature_dim, config.model.num_classes)
        )

        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.accuracy_function = BinaryAccuracy()
        # self.f1_score_function = BinaryF1Score()
        metrics = MetricCollection([BinaryAccuracy(), BinaryF1Score()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        scheduler = ReduceLROnPlateau(opt, patience=2, factor=0.1, verbose=True)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

    def forward(self, inputs) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(inputs)

        outputs = self.classifier(features)

        return outputs

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        pred = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
        loss = self.loss_function(outputs, labels)
        # accuracy = self.accuracy_function(pred, labels)
        # update train metrics
        self.train_metrics.update(pred, labels)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def on_train_epoch_end(self):
        epoch_metric = self.train_metrics.compute()
        self.log_dict(epoch_metric, prog_bar=True, logger=True)

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        pred = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
        loss = self.loss_function(outputs, labels)
        # accuracy = self.accuracy_function(pred, labels)
        # f1_score = self.f1_score_function(pred, labels)
        # update validation metrics
        self.val_metrics.update(pred, labels)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def on_validation_epoch_end(self):
        epoch_metric = self.val_metrics.compute()
        self.log_dict(epoch_metric, prog_bar=True, logger=True)

        self.val_metrics.reset()
