from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "plantvillage"


def load_plantvillage():
    return load_from_disk(DATASET_PATH)


def analyze_leaf_overlap(dataset):
    print("=" * 70)
    print("LEAF ID LEAKAGE ANALYSIS")
    print("=" * 70)

    train_leaf_ids = set()
    test_leaf_ids = set()

    for sample in dataset:
        if sample["split"] == "train":
            train_leaf_ids.add(sample["leaf_id"])
        elif sample["split"] == "test":
            test_leaf_ids.add(sample["leaf_id"])

    overlap = train_leaf_ids.intersection(test_leaf_ids)

    print(f"Unique train leaf IDs: {len(train_leaf_ids):,}")
    print(f"Unique test leaf IDs:  {len(test_leaf_ids):,}")
    print(f"Overlapping leaf IDs:  {len(overlap):,}")

    if len(overlap) == 0:
        print("\nNo leaf IDs appear in both train and test.")
        print("The provided split is leaf-safe.")
    else:
        print("\nWARNING: Leaf IDs appear in both train and test.")
        print("This could cause data leakage.")


def analyze_images_per_leaf(dataset):
    print("\n" + "=" * 70)
    print("IMAGES PER LEAF")
    print("=" * 70)

    leaf_counts = Counter(dataset["leaf_id"])

    print(f"Total unique leaves: {len(leaf_counts):,}")
    print(f"Minimum images per leaf: {min(leaf_counts.values())}")
    print(f"Maximum images per leaf: {max(leaf_counts.values())}")

    distribution = Counter(leaf_counts.values())

    print("\nImages per leaf distribution:")

    for images_per_leaf, number_of_leaves in sorted(distribution.items()):
        print(
            f"{images_per_leaf:3d} images/leaf -> "
            f"{number_of_leaves:6,} leaves"
        )


def analyze_leaf_classes(dataset):
    print("\n" + "=" * 70)
    print("LEAF / CLASS CONSISTENCY")
    print("=" * 70)

    leaf_classes = defaultdict(set)

    for leaf_id, class_name in zip(
        dataset["leaf_id"],
        dataset["class_label"],
    ):
        leaf_classes[leaf_id].add(class_name)

    leaves_with_multiple_classes = {
        leaf_id: classes
        for leaf_id, classes in leaf_classes.items()
        if len(classes) > 1
    }

    print(
        f"Leaves belonging to exactly one class: "
        f"{len(leaf_classes) - len(leaves_with_multiple_classes):,}"
    )

    print(
        f"Leaves belonging to multiple classes: "
        f"{len(leaves_with_multiple_classes):,}"
    )

    if leaves_with_multiple_classes:
        print("\nExamples:")

        for leaf_id, classes in list(leaves_with_multiple_classes.items())[:10]:
            print(f"Leaf {leaf_id}: {classes}")


def analyze_split_by_class(dataset):
    print("\n" + "=" * 70)
    print("TRAIN / TEST DISTRIBUTION BY CLASS")
    print("=" * 70)

    class_split_counts = defaultdict(Counter)

    for class_name, split in zip(
        dataset["class_label"],
        dataset["split"],
    ):
        class_split_counts[class_name][split] += 1

    print(
        f"{'Class':45s} | "
        f"{'Train':>7s} | "
        f"{'Test':>7s}"
    )

    print("-" * 70)

    for class_name in sorted(class_split_counts):
        train_count = class_split_counts[class_name]["train"]
        test_count = class_split_counts[class_name]["test"]

        print(
            f"{class_name:45s} | "
            f"{train_count:7,d} | "
            f"{test_count:7,d}"
        )


def main():
    dataset = load_plantvillage()

    print(f"Loaded dataset: {len(dataset):,} images")

    analyze_leaf_overlap(dataset)
    analyze_images_per_leaf(dataset)
    analyze_leaf_classes(dataset)
    analyze_split_by_class(dataset)


if __name__ == "__main__":
    main()