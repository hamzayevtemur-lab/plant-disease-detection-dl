from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms


train_transform = get_train_transforms()

dataset = PlantVillageDataset(
    csv_file="data/splits/train.csv",
    dataset_dir="data/raw/plantvillage",
    transform=train_transform
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("Dataset size:", len(dataset))
print("Number of batches:", len(dataloader))

image, label = dataset[0]

print("Image type:", type(image))
print("Image shape:", image.shape) #type:ignore
print("Image dtype:", image.dtype) #type:ignore
print("Image min:", image.min().item()) #type:ignore
print("Image max:", image.max().item()) #type:ignore
print("Label:", label)

images, labels = next(iter(dataloader))

print("Batch images shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Batch images dtype:", images.dtype)
print("Batch labels dtype:", labels.dtype)