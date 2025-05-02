import lightning as L
import torch

import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    ConfusionMatrix,
    AUROC,
    CohenKappa,
    MetricCollection,
)


# Training Module
class MolecularSubtypeClassifier(L.LightningModule):

    def __init__(
            self,
            backbone,
            num_classes,
            learning_rate=0.001,
            classification_mode="multiclass",
            class_weights=None,
            reduce_lr_on_plateau=True
    ):
        """
        Lightning Module for Molecular Subtype Classification.
        :param backbone: Backbone of the model.
        :param num_classes: Number of classes for classification.
        :param learning_rate: Learning rate for the optimizer.
        :param classification_mode: Classification mode.
        :param class_weights: Class weights for the loss function.
        """
        super().__init__()

        # Init
        self.backbone = backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.classification_mode = classification_mode
        self.class_weights = class_weights
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

        # Loss function
        if self.class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics
        multiclass_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
                "auroc": AUROC(num_classes=num_classes, task="multiclass"),
                "cohen_kappa": CohenKappa(num_classes=num_classes, task="multiclass"),
            }
        )

        if classification_mode == "multiclass":
            self.train_metrics = multiclass_metrics.clone(prefix="train_")
            self.val_metrics = multiclass_metrics.clone(prefix="val_")
            self.test_metrics = multiclass_metrics.clone(prefix="test_")

            self.train_cm = ConfusionMatrix(num_classes=num_classes, task="multiclass")
            self.val_cm = ConfusionMatrix(num_classes=num_classes, task="multiclass")
            self.test_cm = ConfusionMatrix(num_classes=num_classes, task="multiclass")

        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, x):
        """
        Forward pass through the model.
        :param x: Input data.
        :return: Model output.
        """
        return self.backbone(x)

    def _shared_step(self, batch, batch_idx, stage):
        """
        Shared step for training, validation, and testing.
        :param batch: Input data and labels.
        :param batch_idx: Batch index.
        :param stage: Stage of the training (train/val/test).
        :return: Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        if self.classification_mode == "multiclass":
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            if stage == "val":
                self.val_preds.append(preds.cpu())
                self.val_targets.append(y.cpu())
            elif stage == 'test':
                self.test_preds.append(preds.cpu())
                self.test_targets.append(y.cpu())

            metrics = getattr(self, f"{stage}_metrics")
            for name, metric in metrics.items():
                if "auroc" in name:
                    metric.update(probs, y)
                else:
                    metric.update(preds, y)

            cm = getattr(self, f"{stage}_cm")
            cm.update(preds, y)

        self.log(f"{stage}_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        :param batch: Input data and labels.
        :param batch_idx: Batch index.
        :return: Loss value.
        """
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        :param batch: Input data and labels.
        :param batch_idx: Batch index.
        :return: Loss value.
        """
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        :param batch: Input data and labels.
        :param batch_idx: Batch index.
        :return: Loss value.
        """
        return self._shared_step(batch, batch_idx, "test")

    def _shared_epoch_end(self, stage):
        """
        Shared epoch end for training, validation, and testing.
        :param stage: Stage of the training (train/val/test).
        """
        metrics = getattr(self, f"{stage}_metrics")
        cm = getattr(self, f"{stage}_cm")
        computed = metrics.compute()
        self.log_dict(computed, prog_bar=False, on_epoch=True, on_step=False)
        self._format_metrics_row(stage=stage, computed_metrics=computed)
        metrics.reset()
        cm.reset()

    def _format_metrics_row(self, stage="train", computed_metrics=None):
        """
        Format the metrics for printing.
        :param stage: Stage of the training (train/val/test).
        :return: Formatted string with metrics.
        """
        current_epoch = self.current_epoch + 1
        metrics = computed_metrics
        filtered = {
            k: v.item()
            for k, v in metrics.items()
            if stage in k and isinstance(v, torch.Tensor)
        }
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in filtered.items())
        emoji_per_stage = {
            "train": "🚀",
            "val": "📈",
            "test": "🏆",
        }
        print(f"{emoji_per_stage[stage]} Epoch {current_epoch} [{stage.upper()}] → {metrics_str}")
        print("-" * 50)

    def on_train_epoch_end(self):
        """
        Log metrics at the end of the training epoch.
        """
        self._shared_epoch_end("train")

    def on_validation_epoch_end(self):
        """
        Log metrics at the end of the validation epoch.
        """
        self._shared_epoch_end("val")

        if self.logger is not None and isinstance(self.logger, WandbLogger):
            preds = torch.cat(self.val_preds).tolist()
            targets = torch.cat(self.val_targets).tolist()

            self.logger.experiment.log({
                "val_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=targets,
                    preds=preds,
                    class_names=[f"Class_{i}" for i in range(self.num_classes)]
                )
            })

            self.val_preds.clear()
            self.val_targets.clear()

    def on_test_epoch_end(self):
        """
        Log metrics at the end of the test epoch.
        """
        self._shared_epoch_end("test")

        if self.logger is not None and isinstance(self.logger, WandbLogger):
            preds = torch.cat(self.test_preds).tolist()
            targets = torch.cat(self.test_targets).tolist()

            self.logger.experiment.log({
                "test_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=targets,
                    preds=preds,
                    class_names=[f"Class_{i}" for i in range(self.num_classes)]
                )
            })

            self.test_preds.clear()
            self.test_targets.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate schedulers.
        :return: Optimizer and learning rate scheduler.
        """
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01)
        if self.reduce_lr_on_plateau:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=10, eta_min=1e-6
                ),
            }
        return [optimizer], [scheduler]
