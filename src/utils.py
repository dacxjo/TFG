import os
import shutil

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.config import SPLIT_TRAIN_DIR, SPLIT_VAL_DIR, SPLIT_TEST_DIR


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


def persist_splits(train_df, val_df, test_df, patient_col="patientId", subtype_col="subtype", path_col="convertedPath"):
    train_image_path = SPLIT_TRAIN_DIR
    val_image_path = SPLIT_VAL_DIR
    test_image_path = SPLIT_TEST_DIR

    if os.path.exists(train_image_path):
        shutil.rmtree(train_image_path)

    if os.path.exists(val_image_path):
         shutil.rmtree(val_image_path)

    if os.path.exists(test_image_path):
        shutil.rmtree(test_image_path)

    os.makedirs(train_image_path, exist_ok=True)
    #os.makedirs(val_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)

    copy_images(train_df, train_image_path, path_col, patient_col, subtype_col)
    #copy_images(val_df, val_image_path, path_col, patient_col, subtype_col)
    copy_images(test_df, test_image_path, path_col, patient_col, subtype_col)



def make_grouped_holdout_split(dataframe, patient_col, subtype_col, test_size=0.2):
    patient_subtypes = dataframe[[patient_col, subtype_col]].drop_duplicates()
    trainval_patients, test_patients = train_test_split(
        patient_subtypes, test_size=test_size,
        stratify=patient_subtypes[subtype_col], random_state=42
    )
    trainval_df = dataframe[dataframe[patient_col].isin(trainval_patients[patient_col])]
    test_df = dataframe[dataframe[patient_col].isin(test_patients[patient_col])]
    return trainval_df, test_df

