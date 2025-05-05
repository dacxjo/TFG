import os

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler

from config import CHECKPOINTS_DIR
from src.modeling.classifier import MolecularSubtypeClassifier
from src.modeling.datasets import MammoDataset
from src.modeling.transforms import get_transforms
from src.utils import get_experiment_name, get_class_weights, get_sample_weights, stratified_split


def get_k_folds(
        dataset, n_splits=5, batch_size=64, augment=False, seed=42, oversample=False
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = list(range(len(dataset)))
    y = dataset.labels

    train_transform_fn = lambda: get_transforms(augment=augment)
    val_transform_fn = lambda: get_transforms(augment=False)

    num_workers = min(8, max(1, os.cpu_count() // 2))
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        assert set(train_idx).isdisjoint(set(val_idx)), "Train and Val indices overlap!"

        train_dataset = MammoDataset(
            target_dir=dataset.target_dir,
            extension=dataset.extension,
            transform=train_transform_fn(),
            paths=[dataset.paths[i] for i in train_idx],
        )

        val_dataset = MammoDataset(
            target_dir=dataset.target_dir,
            extension=dataset.extension,
            transform=val_transform_fn(),
            paths=[dataset.paths[i] for i in val_idx],
        )

        assert len(train_dataset) + len(val_dataset) == len(
            dataset
        ), "Mismatch in total samples across folds."
        assert (
                train_dataset.transform is not val_dataset.transform
        ), "Transform contamination!"

        if oversample:
            train_targets = np.array(dataset.labels)[train_idx]
            sample_weights = get_sample_weights(train_targets)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=shuffle, sampler=sampler, **loader_args
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, **loader_args
        )

        yield fold, train_loader, val_loader


def run_kfold_training(
        model_fn,
        model_name,
        dataset,
        n_splits=5,
        augment=False,
        oversample=False,
        experiment_config=None,
        num_classes=4,
        epochs=10,
        learning_rate=0.001,
        device=None,
        enable_logging=False,
):
    try:
        print("Starting K-Fold training...")
        exp_name = get_experiment_name(model=model_name)

        f1_macro_scores = []
        recall_macro_scores = []
        precision_macro_scores = []

        f1_weighted_scores = []
        recall_weighted_scores = []
        precision_weighted_scores = []

        auroc_scores = []
        accuracy_scores = []
        cohen_kappas = []

        logger = None

        for fold, train_loader, val_loader in get_k_folds(
                dataset, n_splits, augment=augment, oversample=oversample
        ):
            print(f"Fold {fold + 1}/{n_splits}")
            print("-" * 50)

            if enable_logging:
                logger = WandbLogger(
                    project="project-aletheia",
                    name=f"{exp_name}-fold{fold}",
                    log_model=False,
                )

            if logger is not None and experiment_config is not None:
                logger.experiment.config.update(experiment_config)

            backbone = model_fn(freeze_backbone=True)

            train_targets = np.array(dataset.labels)
            class_weights = get_class_weights(train_targets).to(device)

            classifier = MolecularSubtypeClassifier(
                backbone=backbone,
                num_classes=num_classes,
                class_weights=class_weights if oversample == False else None,
                class_names=dataset.classes,
                learning_rate=learning_rate,
            )

            checkpoint_cb = ModelCheckpoint(
                dirpath=f"{CHECKPOINTS_DIR}/{exp_name}/",
                monitor="val_auroc", # TODO: Review
                mode="max",
                save_top_k=2,
                filename=f"fold{fold}-{{epoch}}-{{val_auroc:.4f}}",
                save_last=True,
                verbose=True,
            )

            early_stopping_cb = EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
                verbose=True,
                min_delta=0.0001,
            )

            lr_scheduler_cb = LearningRateMonitor(logging_interval="epoch")

            trainer = L.Trainer(
                max_epochs=epochs,
                logger=logger,
                accelerator="auto",
                # precision='16-mixed',
                enable_model_summary=False,
                log_every_n_steps=20,
                callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb],
            )

            trainer.fit(classifier, train_loader, val_loader)

            metrics = trainer.callback_metrics

            f1_macro_scores.append(metrics["val_f1_macro"].item())
            recall_macro_scores.append(metrics["val_recall_macro"].item())
            precision_macro_scores.append(metrics["val_precision_macro"].item())

            f1_weighted_scores.append(metrics["val_f1_weighted"].item())
            recall_weighted_scores.append(metrics["val_recall_weighted"].item())
            precision_weighted_scores.append(metrics["val_precision_weighted"].item())

            auroc_scores.append(metrics["val_auroc"].item())
            accuracy_scores.append(metrics["val_accuracy"].item())
            cohen_kappas.append(metrics["val_cohen_kappa"].item())

            wandb.finish()
            torch.cuda.empty_cache()

        print("\n" + "-" * 100)
        print("Final K-Fold Results:")
        print(
            f"Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}"
        )
        print(f"AUROC: {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")

        print(
            f"F1-Score (Macro): {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}"
        )
        print(
            f"Precision (Macro): {np.mean(precision_macro_scores):.4f} ± {np.std(precision_macro_scores):.4f}"
        )
        print(
            f"Recall (Macro): {np.mean(recall_macro_scores):.4f} ± {np.std(recall_macro_scores):.4f}"
        )

        print(
            f"F1-Score (Weighted): {np.mean(f1_weighted_scores):.4f} ± {np.std(f1_weighted_scores):.4f}"
        )
        print(
            f"Precision (Weighted): {np.mean(precision_weighted_scores):.4f} ± {np.std(precision_weighted_scores):.4f}"
        )
        print(
            f"Recall (Weighted): {np.mean(recall_weighted_scores):.4f} ± {np.std(recall_weighted_scores):.4f}"
        )

        print(f"Cohen Kappa: {np.mean(cohen_kappas):.4f} ± {np.std(cohen_kappas):.4f}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        print(e.with_traceback())
        wandb.finish()
        torch.cuda.empty_cache()

def run_full_training(
        model_fn,
        dataset,
        model_name="resnet101",
        val_split=0.1,
        total_epochs=30,
        freeze_epochs=10,
        fine_tune=True,
        augment=True,
        batch_size=64,
        oversample=False,
        num_classes=4,
        lr=0.001,
        enable_logging=False,
):

    print("Starting full training...")

    train_idx, val_idx = stratified_split(dataset, val_split)

    train_transform_fn = lambda: get_transforms(augment=augment)
    val_transform_fn = lambda: get_transforms(augment=False)

    # Create new datasets for training and validation (validation at least 10% to avoid overfitting)
    train_dataset = MammoDataset(
        target_dir=dataset.target_dir,
        extension=dataset.extension,
        transform=train_transform_fn(),
        paths=[dataset.paths[i] for i in train_idx],
    )

    val_dataset = MammoDataset(
        target_dir=dataset.target_dir,
        extension=dataset.extension,
        transform=val_transform_fn(),
        paths=[dataset.paths[i] for i in val_idx],
    )

    train_labels = np.array(dataset.labels)[train_idx]

    num_workers = min(8, os.cpu_count() // 2)
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if oversample:
        sample_weights = get_sample_weights(train_labels)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        **loader_args
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args
    )

    exp_name = get_experiment_name(model=model_name)

    logger = None
    if enable_logging:
        logger = WandbLogger(name=f"{exp_name}-holdout", log_model=False)

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{CHECKPOINTS_DIR}/{exp_name}/",
        monitor="val_auroc",  # TODO: Review
        mode='max',
        save_top_k=2,
        filename=f"{model_name}-{{epoch}}-{{val_auroc:.4f}}",
        save_last=True,
        verbose=True
    )

    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True,
        min_delta=0.0001
    )

    lr_scheduler_cb = LearningRateMonitor(logging_interval='epoch')

    # Starting with freezed backbone
    backbone = model_fn(freeze_backbone=True)

    classifier = MolecularSubtypeClassifier(
        backbone=backbone,
        num_classes=num_classes,
        class_names=dataset.classes,
        learning_rate=lr
    )

    trainer = L.Trainer(
        max_epochs=freeze_epochs,
        logger=logger,
        accelerator="auto",
        #precision='16-mixed',
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb],
        log_every_n_steps=20
    )

    trainer.fit(classifier, train_loader, val_loader)

    if fine_tune and not trainer.early_stopping_callback.stopped_epoch:
        freeze_checkpoint_path = checkpoint_cb.best_model_path

        print(f"Unfreezing model and resuming training from {freeze_checkpoint_path}")

        classifier = MolecularSubtypeClassifier.load_from_checkpoint(
            freeze_checkpoint_path,
            backbone=model_fn(freeze_backbone=False),
            num_classes=num_classes,
            class_names=dataset.classes,
            learning_rate=1e-5 # Lower learning rate for fine-tuning
        )

        trainer_ft = L.Trainer(
            max_epochs=total_epochs - freeze_epochs,
            logger=logger,
            accelerator="auto",
            #precision='16-mixed',
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb],
            log_every_n_steps=20
        )
        trainer_ft.fit(classifier, train_loader, val_loader)


    if logger is not None:
        wandb.finish()
    torch.cuda.empty_cache()

    best_path = checkpoint_cb.best_model_path
    best_classifier = MolecularSubtypeClassifier.load_from_checkpoint(
        best_path,
        backbone=model_fn(freeze_backbone=False),
        num_classes=num_classes,
        class_names=dataset.classes,
    )
    return best_classifier







