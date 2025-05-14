import os
import pathlib
from collections import Counter

import cv2
import torch
from torchvision import transforms


class MammoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            target_dir: str,
            extension="png",
            transform: transforms = None,
            paths = None
    ):
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
        self.paths = paths if paths is not None else self._load_paths()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self._load_image(idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            img = self.transform(image=img)["image"]
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
            class_counts = dict(
                sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
            )

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
        classes = sorted(
            entry.name for entry in os.scandir(self.target_dir) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {self.target_dir}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_paths(self):
        """
        Load the paths of the images from the directory structure.
        :return: None
        """
        tmp = list(pathlib.Path(self.target_dir).glob(f"*/*.{self.extension}"))
        if len(tmp) == 0:
            raise FileNotFoundError(f"Couldn't find any images in {self.target_dir}.")
        else:
            print(f"Found {len(tmp)} images for {len(self.classes)} classes.")
        return tmp

    @property
    def labels(self):
        """
        Extract class indices from paths.
        :return: List of class indices for each sample.
        """
        return [self.class_to_idx[path.parent.name] for path in self.paths]


class MammoDatasetV2(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(self.dataframe["subtype"].unique())}

    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        mlo_img = cv2.imread(row['mlo_image'], cv2.IMREAD_COLOR)
        mlo_img = cv2.cvtColor(mlo_img, cv2.COLOR_BGR2RGB)
        cc_img = cv2.imread(row['cc_image'], cv2.IMREAD_COLOR)
        cc_img = cv2.cvtColor(cc_img, cv2.COLOR_BGR2RGB)
        label = self.label_to_idx[row["subtype"]]
        if self.transform:
            mlo_img = self.transform(image=mlo_img)["image"]
            cc_img = self.transform(image=cc_img)["image"]

        return (mlo_img, cc_img), label

    def get_image_count_per_class(self, as_percentage=False, sort=True):
        """
        Count the number of images per class.
        :return: Dictionary with class names as keys and number of images as values.
        """
        class_counts = self.dataframe['subtype'].value_counts(normalize=as_percentage)*2
        if sort:
            class_counts = class_counts.sort_values(ascending=False)
        return class_counts.to_dict()

    def find_classes(self):
        classes = sorted(self.dataframe["subtype"].unique())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @property
    def classes(self):
        """
        Get the list of classes.
        :return: List of class names.
        """
        return self.dataframe["subtype"].unique()

    @property
    def labels(self):
        """
        Extract class indices from paths.
        :return: List of class indices for each sample.
        """
        return [self.label_to_idx[label] for label in self.dataframe["subtype"]]
