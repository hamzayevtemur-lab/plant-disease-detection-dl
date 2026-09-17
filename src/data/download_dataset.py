from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "geraldmc/plantvillage-full"
DATASET_REVISION = "v0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage"


def download_dataset():
    print("Loading PlantVillage dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split="train",
    )

    print(f"Dataset loaded: {len(dataset):,} images")

    print(f"\nSaving dataset to:")
    print(RAW_DATA_DIR)

    dataset.save_to_disk(RAW_DATA_DIR)

    print("\nDataset saved successfully.")


if __name__ == "__main__":
    download_dataset()