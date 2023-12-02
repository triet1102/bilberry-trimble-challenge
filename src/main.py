import timm
import torch
from torchmetrics.classification import MulticlassAccuracy
from src.data_preprocessing.data_module import FieldRoadDatasetKFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

writer = SummaryWriter()

logger = wandb.init(
    project="image_classification",
    config={
        "lr": 1e-2,
        "nb_folds": 3,
        "nb_epochs": 5,
        "optimizer": "AdamW",
        "freeze_backbone": True,
    },
)

model = timm.create_model(
    "convnextv2_tiny.fcmae",
    pretrained=True,
    num_classes=2,  # remove classifier nn.Linear
)
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
nb_epochs = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_function = torch.nn.CrossEntropyLoss()
accuracy_function = MulticlassAccuracy(num_classes=2)
softmax = torch.nn.Softmax(dim=-1)

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=True)

dataset_k_fold = FieldRoadDatasetKFold(transforms=transforms, nb_folds=3, batch_size=16)
dataset_k_fold.setup()

for idx in tqdm(range(dataset_k_fold.nb_folds), total=dataset_k_fold.nb_folds):
    for epoch in tqdm(range(nb_epochs), total=nb_epochs):
        epoch_loss = 0
        train_loader = dataset_k_fold.train_dataloader(idx)
        val_loader = dataset_k_fold.val_dataloader(idx)

        # train
        for batch, (data, label) in enumerate(train_loader):
            output = model(data)

            loss = loss_function(output, label)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log train loss, train accuracy
            pred = torch.argmax(softmax(output), dim=-1)
            accuracy = accuracy_function(pred, label)
            writer.add_scalar(
                f"fold_{idx}_epoch_{epoch}_train_loss", loss.item(), batch
            )
            writer.add_scalar(
                f"fold_{idx}_epoch_{epoch}_train_acc", accuracy.item(), batch
            )

            logger.log(
                {
                    f"train/loss/fold_{idx}/epoch_{epoch}": loss.item(),
                    f"train/acc/fold_{idx}_epoch_{epoch}": accuracy.item(),
                }
            )

        # validation
        if epoch % 1 == 0:
            model.eval()
            for batch, (data, label) in enumerate(val_loader):
                with torch.no_grad():
                    output = model(data)
                    loss = loss_function(output, label)

                # log train loss, train accuracy
                pred = torch.argmax(softmax(output), dim=-1)
                accuracy = accuracy_function(pred, label)
                writer.add_scalar(
                    f"fold_{idx}_epoch_{epoch}_val_loss", loss.item(), batch
                )
                writer.add_scalar(
                    f"fold_{idx}_epoch_{epoch}_val_acc", accuracy.item(), batch
                )

                logger.log(
                    {
                        f"val/loss/fold_{idx}/epoch_{epoch}": loss.item(),
                        f"val/acc/fold_{idx}_epoch_{epoch}": accuracy.item(),
                    }
                )

            model.train()

logger.finish()

writer.flush()
writer.close()
print("Finished")
