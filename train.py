import argparse

import lightning as L
import wandb
from dotenv import load_dotenv

from src.modeling.backbones import resnet101_backbone, maxvit_backbone
from src.modeling.datasets import MammoDataset
from src.training.engine import run_kfold_training
from src.utils import get_device
from config import SPLIT_TRAIN_DIR


def validate_args(args):
    if args.backbone not in ["resnet101", "maxvit"]:
        raise ValueError("Invalid model architecture. Choose 'resnet101' or 'maxvit'.")
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


def main():
    load_dotenv()
    print("Starting training...")
    parser = argparse.ArgumentParser(description="Train a model for breast cancer subtype classification.")
    parser.add_argument("--backbone", type=str, default="resnet101", choices=["resnet101", "maxvit"],
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
    args = parser.parse_args()

    validate_args(args)

    SEED = args.seed
    L.seed_everything(SEED)

    # WandB
    wandb.login()

    dataset = MammoDataset(target_dir=args.train_dir, extension=args.extension)

    # Default settings
    model_fn = resnet101_backbone
    model_name = "ResNet101"
    num_classes = 4

    if args.backbone == "resnet101":
        model_fn = resnet101_backbone
        model_name = "ResNet101"
    elif args.backbone == "maxvit":
        model_fn = maxvit_backbone
        model_name = "MaxViT"

    device = get_device()

    training_mode = args.training_mode

    k_folds = args.k_folds

    if training_mode == 'complete':
        print("Full training on the dataset")
    else:
        run_kfold_training(
            model_fn=model_fn,
            model_name=model_name,
            dataset=dataset,
            n_splits=k_folds,
            augment=args.augment,
            oversample=args.oversample,
            num_classes=num_classes,
            epochs=args.epochs,
            device=device,
        )


if __name__ == "__main__":
    main()
