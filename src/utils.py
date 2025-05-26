import os
import random
import shutil
from collections import Counter
import cv2


import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from src.modeling.backbones import resnet101_backbone, maxvit_backbone, vit_backbone, swin_backbone

from config import SPLITS_DATA_DIR
from src.preprocessing import histogram_standarization


def make_grouped_splits(dataframe: DataFrame, patient_col: str, subtype_col: str):
    patient_subtypes = dataframe[[patient_col, subtype_col]].drop_duplicates()

    train_patients, temp_patients = train_test_split(patient_subtypes, test_size=0.3,
                                                     stratify=patient_subtypes[subtype_col], random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.33, stratify=temp_patients[subtype_col],
                                                   random_state=42)

    train_df = dataframe[dataframe[patient_col].isin(train_patients[patient_col])]
    val_df = dataframe[dataframe[patient_col].isin(val_patients[patient_col])]
    test_df = dataframe[dataframe[patient_col].isin(test_patients[patient_col])]

    print(f'Original dataset size: {dataframe.shape[0]}')
    print(f"Train set size: {len(train_df)} ({len(train_df) / len(dataframe):.2%})")
    print(f"Validation set size: {len(val_df)} ({len(val_df) / len(dataframe):.2%})")
    print(f"Test set size: {len(test_df)} ({len(test_df) / len(dataframe):.2%})")

    print("\nSubtype distribution")

    for set_name, dataset in [('Original', dataframe), ('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{set_name} set:")
        print(dataset[subtype_col].value_counts(normalize=True))

    train_patients_set = set(train_df[patient_col].unique())
    val_patients_set = set(val_df[patient_col].unique())
    test_patients_set = set(test_df[patient_col].unique())

    print("\nChecking patient overlap:")
    print(
        f"Train-Val overlap: {len(train_patients_set.intersection(val_patients_set))}")
    print(
        f"Train-Test overlap: {len(train_patients_set.intersection(test_patients_set))}")
    print(
        f"Val-Test overlap: {len(val_patients_set.intersection(test_patients_set))}")

    return train_df, val_df, test_df


def copy_images(df, destination_path, path_col, patient_col, subtype_col):
    for _, row in df.iterrows():
        patient_id = row[patient_col]
        p = row[path_col]
        # make subtype folder if it doesn't exist
        if not os.path.exists(os.path.join(destination_path, row[subtype_col])):
            os.makedirs(os.path.join(destination_path, row[subtype_col]))
        image_name = f"{patient_id}_{p.split('/')[-1]}"
        shutil.copy(p, os.path.join(destination_path, row[subtype_col], image_name))


def copy_images_and_standardize(df, destination_path, path_col, patient_col, subtype_col, standarization_landmarks=None):
    df = df.copy()
    for idx, row in df.iterrows():
        patient_id = row[patient_col]
        p = row[path_col]
        if not os.path.exists(os.path.join(destination_path, row[subtype_col])):
            os.makedirs(os.path.join(destination_path, row[subtype_col]))
        image_name = f"{patient_id}_{p.split('/')[-1]}"
        final_npy_path = os.path.join(destination_path, row[subtype_col], f"{image_name}.npy")

        df.loc[idx, 'img_path'] = final_npy_path
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if standarization_landmarks is not None:
            img = histogram_standarization(img, standarization_landmarks)

        np.save(final_npy_path, img)
    df = df.drop(columns=[path_col])
    df.to_csv(os.path.join(destination_path, 'df.csv'), index=False)



def persist_splits(train_df, val_df, test_df, patient_col="patientId", subtype_col="subtype", path_col="convertedPath", standardize=False, standarization_landmarks=None, seed=42):
    train_image_path = os.path.join(SPLITS_DATA_DIR, str(seed), "train")
    val_image_path = os.path.join(SPLITS_DATA_DIR, str(seed), "val")
    test_image_path = os.path.join(SPLITS_DATA_DIR, str(seed), "test")

    if os.path.exists(train_image_path):
        shutil.rmtree(train_image_path)

    if os.path.exists(val_image_path):
        shutil.rmtree(val_image_path)

    if os.path.exists(test_image_path):
        shutil.rmtree(test_image_path)

    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)

    if standardize:
        copy_images_and_standardize(train_df, train_image_path, path_col, patient_col, subtype_col, standarization_landmarks)
        copy_images_and_standardize(test_df, test_image_path, path_col, patient_col, subtype_col, standarization_landmarks)

        if val_df is not None:
            os.makedirs(val_image_path, exist_ok=True)
            copy_images_and_standardize(val_df, val_image_path, path_col, patient_col, subtype_col, standarization_landmarks)
    else:
        copy_images(train_df, train_image_path, path_col, patient_col, subtype_col)
        copy_images(test_df, test_image_path, path_col, patient_col, subtype_col)

        if val_df is not None:
            os.makedirs(val_image_path, exist_ok=True)
            copy_images(val_df, val_image_path, path_col, patient_col, subtype_col)


def make_grouped_holdout_split(dataframe, patient_col, subtype_col, test_size=0.2, seed=42):
    patient_subtypes = dataframe[[patient_col, subtype_col]].drop_duplicates()
    trainval_patients, test_patients = train_test_split(
        patient_subtypes, test_size=test_size,
        stratify=patient_subtypes[subtype_col], random_state=seed
    )
    trainval_df = dataframe[dataframe[patient_col].isin(trainval_patients[patient_col])]
    test_df = dataframe[dataframe[patient_col].isin(test_patients[patient_col])]
    assert set(trainval_df[patient_col]).isdisjoint(set(test_df[patient_col])), "Patient leakage detected!"
    return trainval_df, test_df


def get_experiment_name(prefix="ALE", model="resnet101"):
    number = random.randint(1, 999)
    return f"{prefix}{number}-{model}"


def get_sample_weights(labels):
    class_counts = Counter(labels)
    total_count = sum(class_counts.values())

    class_weights = {cls: total_count / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return torch.DoubleTensor(sample_weights)


def get_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_backbone_model(backbone_name):
    if backbone_name == "resnet101":
        model_fn = resnet101_backbone
        model_name = "ResNet101"
    elif backbone_name == "maxvit":
        model_fn = maxvit_backbone
        model_name = "MaxVit"
    elif backbone_name == "vit":
        model_fn = vit_backbone
        model_name = "ViT"
    elif backbone_name == "swin":
        model_fn = swin_backbone
        model_name = "Swin"
    else:
        raise ValueError(f"Unknown backbone model: {backbone_name}")

    return model_fn, model_name


def stratified_split(dataset, val_split=0.1):
    indices = list(range(len(dataset)))
    y = dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    for train_idx, val_idx in sss.split(indices, y):
        return train_idx, val_idx
