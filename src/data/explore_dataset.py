from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "plantvillage"


def load_plantvillage():
    return load_from_disk(DATASET_PATH)


def print_dataset_info(dataset):
    print("=" * 70)
    print("PLANTVILLAGE DATASET")
    print("=" * 70)

    print(f"Total images: {len(dataset):,}")
    print(f"Number of features: {len(dataset.features)}")

    print("\nFeatures:")
    for feature in dataset.features:
        print(f"  - {feature}")


def print_class_distribution(dataset):
    class_counts = Counter(dataset["class_label"])

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name:45s} | {count:5d}")

    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Smallest class: {min(class_counts.values()):,} images")
    print(f"Largest class:  {max(class_counts.values()):,} images")


def print_split_distribution(dataset):
    split_counts = Counter(dataset["split"])

    print("\n" + "=" * 70)
    print("PROVIDED SPLIT DISTRIBUTION")
    print("=" * 70)

    for split_name, count in sorted(split_counts.items()):
        print(f"{str(split_name):10s} | {count:6,}")

    print(f"\nTotal: {sum(split_counts.values()):,}")


def print_leaf_information(dataset):
    leaf_ids = dataset["leaf_id"]
    leaf_grouped = dataset["leaf_grouped"]

    print("\n" + "=" * 70)
    print("LEAF INFORMATION")
    print("=" * 70)

    print(f"Total images: {len(dataset):,}")
    print(f"Unique leaf IDs: {len(set(leaf_ids)):,}")

    grouped_counts = Counter(leaf_grouped)

    print("\nleaf_grouped values:")

    for value, count in sorted(grouped_counts.items(), key=lambda x: str(x[0])):
        print(f"  {value}: {count:,}")


def show_sample_images(dataset):
    indices = [
        0,
        100,
        500,
        1000,
        5000,
        10000,
        20000,
        30000,
        40000,
        50000,
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for ax, index in zip(axes.flat, indices):
        sample = dataset[index]

        ax.imshow(sample["image"])

        ax.set_title(
            sample["class_label"],
            fontsize=9,
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    dataset = load_plantvillage()

    print_dataset_info(dataset)
    print_class_distribution(dataset)
    print_split_distribution(dataset)
    print_leaf_information(dataset)

    show_sample_images(dataset)


if __name__ == "__main__":
    main()