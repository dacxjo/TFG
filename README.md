# Breast Cancer Molecular Subtype Classification

## Table of contents
- [Overview](#overview)
- [Project structure](#project-structure)
- [Requirements](#requirements)
  - [Main dependencies](#main-dependencies)
- [Training](#training)
- [Testing](#testing)
- [Prediction](#prediction)


## Overview of the project

This project aims to train a deep learning model to classify breast cancer molecular subtypes using  The Chinese Mammography Database (CMMD).

REF: [CMMD](https://www.cancerimagingarchive.net/collection/cmmd/)

The approach is to try several backbones from the Vision Transformers family, including base Vision Transformer, Swin Transformer, and Multi-Axis Vision Transformer and also ResNet and ResNet101 as baseline.

## Project structure

```
├── 📁 data
│   ├── raw - Raw data, unprocessed images
│   ├── processed - Processed data, ready for training or splitting.
│   └── splits - Data splits for training, testing or validation.
├── 📁 notebooks - Jupyter notebooks for data exploration and visualization.
├── 📁 models 
│   ├── checkpoints - Checkpoints for the trained models.
│   ├── artifacts - Compiled artifacts for the trained models.
├── 📁 src
│   ├── modeling -  Datasets, transforms, and model backbones.
│   ├── training - Utility functions for training.
│   ├── testing - Utility functions for testing.
│   ├── utils - Utility functions for data processing and visualization.
│   └── config.py - Configuration file for the project.
├── 🐍 train.py -  Training entry point.
├── 🐍 test.py -  Testing entry point.
├── 🐍 predict.py -  Prediction entry point.
├── ⚙️ .env - Environment variables for the project.
``` 

## Requirements

Poetry is used for dependency management.

To install Poetry, visit [Poetry installation](https://python-poetry.org/docs/#installation) and follow the instructions for your operating system.

### Main dependencies
- [torch](https://pytorch.org/get-started/locally/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
- [pytorch_lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
- [albumentations](https://albumentations.ai/docs/)

To install the dependencies, run the following command in the root directory of the project:

```bash
poetry install
```

After this, poetry will create a virtual environment for the project and install all the dependencies in it.

## Training

Poetry allows you to run scripts with the virtual environment activated. To run the training script, use the following command:

```bash
poetry run train
```

or

```bash
python -m train
```
in case you want to run the script without poetry.

### Train script arguments

ALLOWED_BACKBONES = ["resnet101", "maxvit", "vit", "swin"]

| Argument | Type | Default Value | Description |
|-----------|------|--------------|-------------|
| `--backbone` | string | "resnet101" | Model architecture to use (options available in ALLOWED_BACKBONES) |
| `--train_dir` | string | SPLIT_TRAIN_DIR | Path to training data directory |
| `--epochs` | int | 10 | Number of epochs to train |
| `--batch_size` | int | 64 | Batch size for training |
| `--augment` | flag | False | Enable to use data augmentation |
| `--oversample` | flag | False | Enable to use oversampling for class imbalance |
| `--training-mode` | string | 'k-folds' | Training mode ('k-folds' or 'full') |
| `--k_folds` | int | 3 | Number of folds for K-Fold cross-validation |
| `--extension` | string | 'png' | Image file extension |
| `--seed` | int | 42 | Seed for random number generation |
| `--num_classes` | int | 4 | Number of classes for classification |
| `--enable_logging` | flag | False | Enable to activate logging with WandB |
| `--lr` | float | 0.001 | Learning rate |

Training can be done using K-Fold cross-validation or full training. The `--training-mode` argument can be set to either `k-folds` or `full`. If `k-folds` is selected, the `--k_folds` argument can be used to specify the number of folds. The training script will create a directory for each fold in the `models/checkpoints` directory.

### Examples

```bash
# Train a model with default parameters
poetry run train
# Train a model with a different backbone
poetry run train --backbone maxvit --epochs 40 --batch_size 32 --augment --enable_logging
```

## Testing

The testing script is used to evaluate the performance of the trained model on a test dataset. It can be run with the following command, passing as argument the path to the model checkpoint:

```bash
poetry run test --checkpoint_dir models/checkpoints/maxvit-codenam-53/fold_0.ckpt --backbone maxvit
```

### Test script arguments

| Argument | Type | Default Value | Description |
|-----------|------|--------------|-------------|
| `--test_dir` | string | SPLIT_TEST_DIR | Path to test data directory |
| `--checkpoint_dir` | string | None (required) | Path to the model checkpoint directory |
| `--batch_size` | int | 32 | Batch size for evaluation |
| `--backbone` | string | "resnet101" | Model architecture to use (options available in ALLOWED_BACKBONES) |
| `--extension` | string | 'png' | Image file extension |
| `--report` | flag | False | Enable to generate a classification report |

### Examples

```bash
# Test a model with default parameters
poetry run test --checkpoint_dir models/checkpoints/maxvit-codenam-53/fold_0.ckpt
# Test a model with a different backbone and classification report
poetry run test --checkpoint_dir models/checkpoints/maxvit-codenam-53/fold_0.ckpt --backbone maxvit --batch_size 16 --report
```

## Prediction

Right now, prediction will load an onnx model and run inference on a single image. The image should be passed as an argument to the script.

The model path can be also passed as an argument, but it will default to the `best.onnx` file in the `models/artifacts` directory.

This script can be run with the following command, passing as argument the path to the image:

```bash
poetry run predict --test_image data/raw/benign/test_image.png
```
### Predict script arguments

| Argument | Type | Default Value | Description |
|-----------|------|--------------|-------------|
| `--test_image` | string | None | Path to the test image |
| `--model_path` | string | None | Path to the ONNX model |

## Other notes

- The project is using [WandB](https://wandb.ai/) for logging and tracking experiments. You can enable it by passing the `--enable_logging` flag to the training script. The WANDB_API_KEY should be set in the `.env` file.

