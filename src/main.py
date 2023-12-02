from src.data_preprocessing.data_module import FieldRoadDatasetKFold

datamodule = FieldRoadDatasetKFold(
    root_dir="dataset/train",
    batch_size=32,
    image_size=224,
    nb_folds=5,
    split_seed=42,
)

datamodule.setup()
print("OK1")

train_loader = datamodule.train_dataloader(fold_index=0)
test_loader = datamodule.val_dataloader(fold_index=0)

for img, label in train_loader:
    print(img.shape, label.shape)


print("OK2")

for img, label in test_loader:
    print(img.shape, label.shape)

print("OK3")
