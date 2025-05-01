import argparse
import os

import lightning as L
import numpy as np
import torch
import wandb

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from src.datasets import MammoDataset
from src.models import MolecularSubtypeClassifier, resnet101_backbone, maxvit_backbone, get_transforms
from src.utils import get_experiment_name, get_class_weights, get_sample_weights
from dotenv import load_dotenv

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


def run_complete_training(model, model_name, dataset, oversample=False, experiment_config=None, num_classes=4,
                          epochs=10, batch_size=64, validation_split=0.05, device=None):
    try:
        # Create experiment name and logger
        exp_name = get_experiment_name(model=model_name)
        wandb_logger = WandbLogger(
            project="project-aletheia",
            name=f"{exp_name}",
            log_model=False
        )

        if experiment_config is not None:
            wandb_logger.experiment.config.update(experiment_config)

        num_workers = min(8, os.cpu_count() // 2)
        loader_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        train_targets = np.array(dataset.labels)

        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        class_weights = get_class_weights(train_targets).to(device)

        # Initialize model
        classifier = MolecularSubtypeClassifier(
            backbone=model,
            num_classes=num_classes,
            class_weights=class_weights
        )

        # DataLoaders
        if oversample:
            sample_weights = get_class_weights(train_targets)
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
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)

        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)

        # Trainer
        trainer = L.Trainer(
            max_epochs=epochs,
            logger=wandb_logger,
            accelerator="auto",
            enable_model_summary=False,
            deterministic=True
        )

        # Training with validation
        trainer.fit(classifier, train_loader, val_loader)

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        wandb.finish()
        torch.cuda.empty_cache()


def main():
    load_dotenv()
    print("Starting training...")
    parser = argparse.ArgumentParser(description="Train a model for breast cancer subtype classification.")
    parser.add_argument("--model", type=str, default="resnet101", choices=["resnet101", "maxvit"],
                        help="Model architecture to use.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation.")
    parser.add_argument("--oversample", action="store_true", help="Use oversampling for class imbalance.")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes for classification.")
    parser.add_argument("--training-mode", type=str, default='k-folds', choices=['k-folds', 'complete'], help="Training mode")
    parser.add_argument('--kfolds', type=int, default=3, help='Number of folds for K-Fold cross-validation')
    args = parser.parse_args()

    # SEED
    SEED = 42
    L.seed_everything(SEED)

    # WandB
    wandb.login()

    dataset = MammoDataset(target_dir="/Users/dacxjo/TFG/data/splits/train", extension='jpg')

    model_fn = resnet101_backbone
    model_name = "ResNet101"

    if args.model == "resnet101":
        model_fn = resnet101_backbone
        model_name = "ResNet101"
    elif args.model == "maxvit":
        model_fn = maxvit_backbone
        model_name = "MaxViT"

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

    training_mode = args.training_mode

    kfolds = args.kfolds

    if training_mode == 'complete':
        run_complete_training(
            model=model_fn(freeze_backbone=True),
            model_name=model_name,
            dataset=dataset,
            oversample=args.oversample,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.05,
            device=device,
        )
    else:
        run_kfold_training(
            model_fn=model_fn,
            model_name=model_name,
            dataset=dataset,
            n_splits=kfolds,
            augment=args.augment,
            oversample=args.oversample,
            num_classes=args.num_classes,
            epochs=args.epochs,
            device=device,

        )

if __name__ == "__main__":
    main()
