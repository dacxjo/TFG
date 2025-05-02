import os

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler

from src.modeling.classifier import MolecularSubtypeClassifier
from src.modeling.transforms import get_transforms
from src.utils import get_experiment_name, get_class_weights, get_sample_weights


def get_k_folds(dataset, n_splits=5, batch_size=64, augment=False, seed=42, oversample=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = list(range(len(dataset)))
    y = dataset.labels

    train_transform = get_transforms(augment)
    val_transform = get_transforms(augment=False)

    num_workers = min(8, os.cpu_count() // 2)
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        if oversample:
            train_targets = np.array(dataset.labels)[train_idx]
            sample_weights = get_sample_weights(train_targets)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                sampler=sampler,
                **loader_args
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                **loader_args
            )
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)

        yield fold, train_loader, val_loader


def run_kfold_training(model_fn, model_name, dataset, n_splits=5, augment=False, oversample=False,
                       experiment_config=None, num_classes=4, epochs=10, device=None):
    try:
        print("Starting K-Fold training...")
        exp_name = get_experiment_name(model=model_name)

        f1_scores = []
        recall_scores = []
        auroc_scores = []
        precision_scores = []
        accuracy_scores = []
        cohen_kappas = []

        for fold, train_loader, val_loader in get_k_folds(dataset, n_splits, augment=augment, oversample=oversample):
            print(f"Fold {fold + 1}/{n_splits}")
            print("-" * 50)

            wandb_logger = WandbLogger(
                project="project-aletheia",
                name=f"{exp_name}-fold{fold}",
                log_model=False
            )

            if experiment_config is not None:
                wandb_logger.experiment.config.update(experiment_config)

            backbone = model_fn(freeze_backbone=True)

            train_targets = np.array(dataset.labels)
            class_weights = get_class_weights(train_targets).to(device)

            classifier = MolecularSubtypeClassifier(
                backbone=backbone,
                num_classes=num_classes,
                class_weights=class_weights,
            )

            checkpoint_cb = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename=f'fold{fold}-{{epoch}}-{{val_loss:.4f}}',
                save_last=True,
                verbose=True
            )

            early_stopping_cb = EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min',
                verbose=True
            )

            trainer = L.Trainer(
                max_epochs=epochs,
                logger=wandb_logger,
                accelerator="auto",
                enable_model_summary=False,
                log_every_n_steps=20,
                callbacks=[checkpoint_cb, early_stopping_cb, RichProgressBar()]
            )

            trainer.fit(classifier, train_loader, val_loader)

            metrics = trainer.callback_metrics

            f1_scores.append(metrics['val_f1'].item())
            recall_scores.append(metrics['val_recall'].item())
            auroc_scores.append(metrics['val_auroc'].item())
            precision_scores.append(metrics['val_precision'].item())
            accuracy_scores.append(metrics['val_accuracy'].item())
            cohen_kappas.append(metrics['val_cohen_kappa'].item())

            wandb.finish()

            torch.cuda.empty_cache()

        print("-" * 100)
        print("Final K-Fold Results:")
        print(f"F1-Score Mean: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Recall Mean: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
        print(f"AUROC Mean: {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
        print(f"Precision Mean: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
        print(f"Accuracy Mean: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        print(f"Cohen Kappa Mean: {np.mean(cohen_kappas):.4f} ± {np.std(cohen_kappas):.4f}")


    except Exception as e:
        print(f"An error occurred during training: {e}")
        print(e.with_traceback())
        wandb.finish()
        torch.cuda.empty_cache()
