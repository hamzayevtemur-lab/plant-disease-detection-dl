from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "plantvillage"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def load_dataset():
    print("Loading local PlantVillage dataset...")

    dataset = load_from_disk(DATASET_PATH)

    print(f"Total images: {len(dataset):,}")

    return dataset


def create_dataframe(dataset):
    print("\nCreating metadata dataframe...")

    dataframe = pd.DataFrame(
        {
            "image_index": range(len(dataset)),
            "class_idx": dataset["class_idx"],
            "class_label": dataset["class_label"],
            "leaf_id": dataset["leaf_id"],
            "split": dataset["split"],
        }
    )

    return dataframe


def split_train_validation(dataframe):
    original_train = dataframe[dataframe["split"] == "train"].copy()

    print("\nOriginal training images:")
    print(f"{len(original_train):,}")

    leaf_metadata = (
        original_train[
            ["leaf_id", "class_idx", "class_label"]
        ]
        .drop_duplicates("leaf_id")
        .reset_index(drop=True)
    )

    print(f"Unique training leaves: {len(leaf_metadata):,}")

    train_leaves, validation_leaves = train_test_split(
        leaf_metadata,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=leaf_metadata["class_idx"],
    )

    train_leaf_ids = set(train_leaves["leaf_id"])
    validation_leaf_ids = set(validation_leaves["leaf_id"])

    train_dataframe = original_train[
        original_train["leaf_id"].isin(train_leaf_ids)
    ].copy()

    validation_dataframe = original_train[
        original_train["leaf_id"].isin(validation_leaf_ids)
    ].copy()

    return train_dataframe, validation_dataframe


def create_test_dataframe(dataframe):
    return dataframe[dataframe["split"] == "test"].copy()


def save_splits(train_dataframe, validation_dataframe, test_dataframe):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "image_index",
        "class_idx",
        "class_label",
        "leaf_id",
    ]

    train_dataframe[columns].to_csv(
        SPLIT_DIR / "train.csv",
        index=False,
    )

    validation_dataframe[columns].to_csv(
        SPLIT_DIR / "validation.csv",
        index=False,
    )

    test_dataframe[columns].to_csv(
        SPLIT_DIR / "test.csv",
        index=False,
    )

    print("\nSplits saved to:")
    print(SPLIT_DIR)


def verify_splits(
    train_dataframe,
    validation_dataframe,
    test_dataframe,
):
    train_leaves = set(train_dataframe["leaf_id"])
    validation_leaves = set(validation_dataframe["leaf_id"])
    test_leaves = set(test_dataframe["leaf_id"])

    train_validation_overlap = train_leaves & validation_leaves
    train_test_overlap = train_leaves & test_leaves
    validation_test_overlap = validation_leaves & test_leaves

    print("\n" + "=" * 70)
    print("SPLIT VERIFICATION")
    print("=" * 70)

    print(f"Train images:      {len(train_dataframe):,}")
    print(f"Validation images: {len(validation_dataframe):,}")
    print(f"Test images:       {len(test_dataframe):,}")

    print()

    print(f"Train leaves:      {len(train_leaves):,}")
    print(f"Validation leaves: {len(validation_leaves):,}")
    print(f"Test leaves:       {len(test_leaves):,}")

    print("\nLeaf overlap:")
    print(f"Train / Validation: {len(train_validation_overlap)}")
    print(f"Train / Test:       {len(train_test_overlap)}")
    print(f"Validation / Test:  {len(validation_test_overlap)}")

    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    ):
        print("\nNo leaf leakage detected.")
    else:
        raise ValueError("Leaf leakage detected!")


def print_class_distribution(
    train_dataframe,
    validation_dataframe,
    test_dataframe,
):
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    train_counts = train_dataframe["class_idx"].value_counts()
    validation_counts = validation_dataframe["class_idx"].value_counts()
    test_counts = test_dataframe["class_idx"].value_counts()

    class_names = (
        train_dataframe[
            ["class_idx", "class_label"]
        ]
        .drop_duplicates()
        .set_index("class_idx")["class_label"]
        .to_dict()
    )

    print(
        f"{'Class':45s} | "
        f"{'Train':>7s} | "
        f"{'Val':>7s} | "
        f"{'Test':>7s}"
    )

    print("-" * 70)

    for class_idx in sorted(class_names):
        print(
            f"{class_names[class_idx]:45s} | "
            f"{train_counts.get(class_idx, 0):7,d} | "
            f"{validation_counts.get(class_idx, 0):7,d} | "
            f"{test_counts.get(class_idx, 0):7,d}"
        )


def main():
    dataset = load_dataset()

    dataframe = create_dataframe(dataset)

    train_dataframe, validation_dataframe = split_train_validation(
        dataframe
    )

    test_dataframe = create_test_dataframe(dataframe)

    verify_splits(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    print_class_distribution(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    save_splits(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


if __name__ == "__main__":
    main()