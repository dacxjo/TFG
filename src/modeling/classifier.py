import lightning as L
import torch

import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, \
    MulticlassAUROC, MulticlassCohenKappa, MulticlassConfusionMatrix


# Training Module
class MolecularSubtypeClassifier(L.LightningModule):

    def __init__(
            self,
            backbone,
            num_classes,
            learning_rate=0.001,
            classification_mode="multiclass",
            class_weights=None,
            class_names=None,
            input_size=(224, 224),
    ):
        """
        Lightning Module for Molecular Subtype Classification.
        :param backbone: Backbone of the model.
        :param num_classes: Number of classes for classification.
        :param learning_rate: Learning rate for the optimizer.
        :param classification_mode: Classification mode.
        :param class_weights: Class weights for the loss function.
        :param class_names: List of class names.
        """
        super().__init__()

        # Init
        self.backbone = backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.classification_mode = classification_mode
        self.class_weights = class_weights
        self.class_names = class_names if class_names is not None else [f"Class_{i}" for i in range(num_classes)]
        self.input_size = input_size

        self.val_preds = []
        self.val_targets = []
        self.val_probs = []
        self.test_preds = []
        self.test_targets = []
        self.test_probs = []

        # Loss function
        if self.class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.05)
        else:
            # Focal Loss may be used here
            self.criterion = torch.nn.CrossEntropyLoss()

        self._init_metrics()

        self.example_input_array = torch.randn(1, 3, *self.input_size)

        self.save_hyperparameters(ignore=["backbone"])

    def _init_metrics(self):
        multiclass_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.num_classes),

                "f1_macro": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
                "precision_macro": MulticlassPrecision(num_classes=self.num_classes, average="macro"),
                "recall_macro": MulticlassRecall(num_classes=self.num_classes, average="macro"),

                "f1_weighted": MulticlassF1Score(num_classes=self.num_classes, average="weighted"),
                "precision_weighted": MulticlassPrecision(num_classes=self.num_classes, average="weighted"),
                "recall_weighted": MulticlassRecall(num_classes=self.num_classes, average="weighted"),

                "auroc": MulticlassAUROC(num_classes=self.num_classes),
                "cohen_kappa": MulticlassCohenKappa(num_classes=self.num_classes)
            }
        )

        if self.classification_mode == "multiclass":
            self.train_metrics = multiclass_metrics.clone(prefix="train_")
            self.val_metrics = multiclass_metrics.clone(prefix="val_")
            self.test_metrics = multiclass_metrics.clone(prefix="test_")

            self.train_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.val_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.test_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)

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
                self.val_probs.append(probs.cpu())
            elif stage == 'test':
                self.test_preds.append(preds.cpu())
                self.test_targets.append(y.cpu())
                self.test_probs.append(probs.cpu())

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
            probs = torch.cat(self.val_probs).tolist()

            self.logger.experiment.log({
                "val_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=targets,
                    preds=preds,
                    class_names=self.class_names
                )
            })

            self.logger.experiment.log({
                "val_roc_curve": wandb.plot.roc_curve(
                    y_true=targets,
                    y_probas=probs,
                    labels=self.class_names
                )
            })

            self.logger.experiment.log({
                "val_pr_curve": wandb.plot.pr_curve(
                    y_true=targets,
                    y_probas=probs,
                    labels=self.class_names
                )
            })

            self.val_preds.clear()
            self.val_targets.clear()
            self.val_probs.clear()

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,
                                                                              eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
