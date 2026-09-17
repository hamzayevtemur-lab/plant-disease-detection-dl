import matplotlib.pyplot as plt
from torchvision import transforms

from src.data.dataset import PlantVillageDataset


IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


dataset = PlantVillageDataset(
    csv_file="data/splits/train.csv",
    dataset_dir="data/raw/plantvillage"
)

image, label = dataset[0]

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
flip = transforms.RandomHorizontalFlip(p=1.0)
rotate = transforms.RandomRotation(15)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD
)

resized = resize(image)
flipped = flip(resized)
rotated = rotate(flipped)
tensor = to_tensor(rotated)
normalized = normalize(tensor)

mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)

denormalized = normalized * std + mean
denormalized = denormalized.clamp(0, 1)

fig, axes = plt.subplots(1, 6, figsize=(18, 4))

axes[0].imshow(image)
axes[0].set_title("Original")

axes[1].imshow(resized)
axes[1].set_title("Resize")

axes[2].imshow(flipped)
axes[2].set_title("Horizontal Flip")

axes[3].imshow(rotated)
axes[3].set_title("Rotation")

axes[4].imshow(tensor.permute(1, 2, 0))
axes[4].set_title("ToTensor")

axes[5].imshow(denormalized.permute(1, 2, 0))
axes[5].set_title("Normalized")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Label:", label)
print("Original size:", image.size) # type: ignore
print("Resized size:", resized.size)
print("Tensor shape:", tensor.shape)
print("Normalized shape:", normalized.shape)