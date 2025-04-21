from collections import Counter

import torch
import os
import pathlib
import cv2
import albumentations as A
from torchvision.transforms import transforms


class MammoDataset(torch.utils.data.Dataset):
    def __init__(self, target_dir: str, extension="png", transform:transforms=None,):
        """
        Custom Dataset for loading the images from a directory structure.
        :param target_dir: Directory path where the images are stored.
        :param transform: Transforms to be applied to the images.
        :param extension: Image extension (default= "png").
        """
        self.target_dir = target_dir
        self.transform = transform
        self.extension = extension
        self.classes, self.class_to_idx = self._find_classes()
        self.paths = self._load_paths()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self._load_image(idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            img = self.transform(img)
        return img, class_idx

    def get_image_count_per_class(self, as_percentage=False, sort=True):
        """
        Count the number of images per class.
        :return: Dictionary with class names as keys and number of images as values.
        """
        class_counts = Counter([path.parent.name for path in self.paths])

        if as_percentage:
            total = len(self.paths)
            class_counts = {k: v / total * 100 for k, v in class_counts.items()}

        if sort:
            class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

        return class_counts

    def _load_image(self, index):
        """
        Load an image from the dataset.
        :param index: Index of the image to load.
        :return: Loaded image.
        """
        img = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _find_classes(self):
        """
        Find the classes from the directory structure.
        :return: None
        """
        classes = sorted(entry.name for entry in os.scandir(self.target_dir) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {self.target_dir}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_paths(self):
        """
        Load the paths of the images from the directory structure.
        :return: None
        """
        tmp = list(pathlib.Path(self.target_dir).glob(f'*/*.{self.extension}'))
        if len(tmp) == 0:
            raise FileNotFoundError(f"Couldn't find any images in {self.target_dir}.")
        else:
            print(f"Found {len(tmp)} images for {len(self.classes)} classes.")
        return tmp


class MolecularDatasetV2(torch.utils.data.Dataset):
    def __init__(self, target_dir: str, transform=None, oversampling_transform=None,
                 img_ext="png", sampling_strategy='original'):
        """
        Molecular Dataset
        :param target_dir: Directory containing the images
        :param transform: Transform to apply to the images
        :param oversampling_transform: Transform to apply for oversampling
        :param img_ext: Image extension
        :param sampling_strategy: Sampling strategy ('original' or 'oversample')
        """
        self.target_dir = target_dir
        self.transform = transform
        self.img_ext = img_ext
        self.sampling_strategy = sampling_strategy

        # Set up oversampling transform
        if oversampling_transform is not None:
            self.oversampling_transform = oversampling_transform
        else:
            self.oversampling_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(std_range=(0.05, 0.2), p=0.2),
                # Add more augmentations as needed
            ])

        # Initialize dataset structure
        self.classes, self.class_to_idx = self._find_classes()
        self.paths = list(pathlib.Path(self.target_dir).glob(f'*/*.{self.img_ext}'))

        if len(self.paths) == 0:
            raise FileNotFoundError(f"Couldn't find any images in {self.target_dir}.")

        # Calculate class distribution
        self.labels = []
        for path in self.paths:
            class_name = path.parent.name
            self.labels.append(self.class_to_idx[class_name])

        # Print original class distribution
        class_counts = Counter(self.labels)
        print(f"Found {len(self.paths)} images for {len(self.classes)} classes.")
        print(f"Original class distribution: {class_counts}")

        # Set up oversampling if needed
        if self.sampling_strategy == 'oversample':
            self._setup_oversampling()

    def _setup_oversampling(self):
        """
        Set up oversampling indices to balance the dataset
        """
        # Calculate class distribution
        class_counts = Counter(self.labels)
        max_count = max(class_counts.values())

        # Create expanded indices for oversampling
        self.expanded_indices = []
        self.apply_augmentation = []

        # First add all original samples
        for i in range(len(self.paths)):
            self.expanded_indices.append(i)
            self.apply_augmentation.append(False)  # Original samples, no augmentation

        # Add augmented samples for minority classes
        for i, label in enumerate(self.labels):
            count = class_counts[label]
            # Add additional augmented samples to match majority class
            if count < max_count:
                repeats_needed = max_count // count - 1
                additional_needed = max_count - count - (repeats_needed * count)

                # Full repeats
                for _ in range(repeats_needed):
                    self.expanded_indices.append(i)
                    self.apply_augmentation.append(True)  # Augmented sample

                # Partial repeat (if needed)
                if additional_needed > 0:
                    self.expanded_indices.append(i)
                    self.apply_augmentation.append(True)

        # Verify new class distribution
        expanded_labels = [self.labels[self.expanded_indices[i]] for i in range(len(self.expanded_indices))]
        balanced_counts = Counter(expanded_labels)
        print(f"After oversampling: {len(self.expanded_indices)} samples")
        print(f"Balanced class distribution: {balanced_counts}")

    def __len__(self):
        if self.sampling_strategy == 'oversample':
            return len(self.expanded_indices)
        return len(self.paths)

    def __getitem__(self, idx):
        if self.sampling_strategy == 'oversample':
            # Get the original index from expanded indices
            original_idx = self.expanded_indices[idx]
            needs_augment = self.apply_augmentation[idx]
        else:
            original_idx = idx
            needs_augment = False

        # Load the image
        img = self.load_image(original_idx)
        class_name = self.paths[original_idx].parent.name
        class_idx = self.class_to_idx[class_name]

        # Apply augmentation if needed (for oversampled images)
        if needs_augment:
            img = self.oversampling_transform(image=img)['image']

        # Apply regular transforms for all images
        if self.transform:
            img = self.transform(image=img)['image']

        return img, class_idx

    def load_image(self, index):
        img = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _find_classes(self):
        classes = sorted(entry.name for entry in os.scandir(self.target_dir) if entry.is_dir())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {self.target_dir}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
