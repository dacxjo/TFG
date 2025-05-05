import argparse

import lightning as L
import wandb
from dotenv import load_dotenv

from src.modeling.datasets import MammoDataset
from src.training.engine import run_kfold_training, run_full_training
from src.utils import get_device, get_backbone_model
from config import SPLIT_TRAIN_DIR, ALLOWED_BACKBONES


def validate_args(args):
    if args.backbone not in ALLOWED_BACKBONES:
        raise ValueError("Invalid model architecture.")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.training_mode not in ['k-folds', 'full']:
        raise ValueError("Invalid training mode. Choose 'k-folds' or 'full'.")
    if args.k_folds <= 0:
        raise ValueError("Number of folds must be a positive integer.")
    if args.extension not in ['jpg', 'png']:
        raise ValueError("Invalid image file extension. Choose 'jpg' or 'png'.")
    if args.seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    if args.num_classes <= 0:
        raise ValueError("Number of classes must be a positive integer.")
    if args.lr <= 0:
        raise ValueError("Learning rate must be a positive float.")


def main():
    load_dotenv()

    print("Starting training...")
    parser = argparse.ArgumentParser(description="Train a model for breast cancer subtype classification.")
    parser.add_argument("--backbone", type=str, default="resnet101", choices=ALLOWED_BACKBONES,
                        help="Model architecture to use.")
    parser.add_argument("--train_dir", type=str, default=SPLIT_TRAIN_DIR, help="Path to training data directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation.")
    parser.add_argument("--oversample", action="store_true", help="Use oversampling for class imbalance.")
    parser.add_argument("--training-mode", type=str, default='k-folds', choices=['k-folds', 'full'],
                        help="Training mode")
    parser.add_argument('--k_folds', type=int, default=3, help='Number of folds for K-Fold cross-validation')
    parser.add_argument("--extension", type=str, default='png', help="Image file extension.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for classification')
    parser.add_argument("--enable_logging", action="store_true", help="Enable logging with WandB.")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    args = parser.parse_args()

    validate_args(args)

    device = get_device()

    SEED = args.seed
    L.seed_everything(SEED)

    if args.enable_logging:
        wandb.login()

    dataset = MammoDataset(target_dir=args.train_dir, extension=args.extension)

    model_fn, model_name = get_backbone_model(args.backbone)

    if args.training_mode == 'full':
        run_full_training(
            model_fn=model_fn,
            model_name=model_name,
            dataset=dataset,
            augment=args.augment,
            oversample=args.oversample,
            num_classes=args.num_classes,
            total_epochs=args.epochs,
            enable_logging=args.enable_logging,
            lr=args.lr,
        )
    else:
        run_kfold_training(
            model_fn=model_fn,
            model_name=model_name,
            dataset=dataset,
            n_splits=args.k_folds,
            augment=args.augment,
            oversample=args.oversample,
            num_classes=args.num_classes,
            epochs=args.epochs,
            enable_logging=args.enable_logging,
            learning_rate=args.lr,
            device=device,
        )


if __name__ == "__main__":
    main()
