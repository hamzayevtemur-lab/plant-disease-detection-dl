from src.data.dataloader import create_dataloaders


train_loader, validation_loader, test_loader = create_dataloaders(
    batch_size=32
)

print("Train dataset size:", len(train_loader.dataset)) # type:ignore
print("Validation dataset size:", len(validation_loader.dataset)) # type:ignore
print("Test dataset size:", len(test_loader.dataset)) # type:ignore

print("Train batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
print("Test batches:", len(test_loader))

train_images, train_labels = next(iter(train_loader))
validation_images, validation_labels = next(iter(validation_loader))
test_images, test_labels = next(iter(test_loader))

print()
print("Train batch:")
print("Images:", train_images.shape)
print("Labels:", train_labels.shape)

print()
print("Validation batch:")
print("Images:", validation_images.shape)
print("Labels:", validation_labels.shape)

print()
print("Test batch:")
print("Images:", test_images.shape)
print("Labels:", test_labels.shape)