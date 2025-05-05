import argparse

from config import SPLIT_TEST_DIR, ALLOWED_BACKBONES
from src.modeling.datasets import MammoDataset
from src.modeling.transforms import get_transforms
from src.testing.engine import test_from_checkpoint
from src.utils import get_backbone_model

def main():
    print("Starting testing...")

    parser = argparse.ArgumentParser(description="Test a model for breast cancer subtype classification.")
    parser.add_argument("--test_dir", type=str, default=SPLIT_TEST_DIR, help="Path to test data directory.")
    parser.add_argument('--checkpoint_dir', type=str, default=None, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--backbone", type=str, default="resnet101", choices=ALLOWED_BACKBONES,
                            help="Model architecture to use.")
    parser.add_argument("--extension", type=str, default='png', help="Image file extension.")
    parser.add_argument("--report", action="store_true", help="Generate a classification report.")

    args = parser.parse_args()

    test_transform = get_transforms(augment=False)

    backbone_fn, _ = get_backbone_model(args.backbone)

    dataset = MammoDataset(
        target_dir=args.test_dir,
        transform=test_transform,
        extension=args.extension,
    )

    test_from_checkpoint(
        dataset=dataset,
        backbone_fn=backbone_fn,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        generate_report=args.report,
    )


if __name__ == "__main__":
    main()