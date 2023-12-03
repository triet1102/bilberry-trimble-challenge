import timm
import torch
from torchmetrics.classification import BinaryAccuracy
from src.data_preprocessing.data_module import FieldRoadDatasetKFold
from tqdm import tqdm
from dataclasses import dataclass
import wandb

torch.manual_seed(42)


@dataclass
class Config:
    lr: float
    nb_folds: int
    nb_epochs: int
    optimizer: torch.optim.Optimizer
    batch_size: int
    freeze_backbone: bool


config = Config(
    lr=1e-3,
    nb_folds=3,
    nb_epochs=20,
    optimizer=torch.optim.AdamW,
    batch_size=32,
    freeze_backbone=True,
)

model = timm.create_model(
    "convnextv2_tiny.fcmae",
    pretrained=True,
    num_classes=1,  # remove classifier nn.Linear
)
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params: {pytorch_total_params}")
nb_epochs = 10

optimizer = config.optimizer(model.parameters(), lr=config.lr)
loss_function = torch.nn.BCEWithLogitsLoss()
accuracy_function = BinaryAccuracy()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms_train = timm.data.create_transform(**data_config, is_training=True, no_aug=True)
transforms_eval = timm.data.create_transform(**data_config, is_training=False)

dataset_k_fold = FieldRoadDatasetKFold(
    transforms_train=transforms_train, transforms_eval=transforms_eval, nb_folds=config.nb_folds, batch_size=config.batch_size
)
dataset_k_fold.setup()
print("OK")

logger = wandb.init(project="image_classification", config=config.__dict__)

# define custom step for wandb logger
logger.define_metric("custom_step")
for idx in range(dataset_k_fold.nb_folds):
    logger.define_metric(f"fold_{idx}/train/loss", step_metric="custom_step")
    logger.define_metric(f"fold_{idx}/train/acc", step_metric="custom_step")
    logger.define_metric(f"fold_{idx}/val/loss", step_metric="custom_step")
    logger.define_metric(f"fold_{idx}/val/acc", step_metric="custom_step")

for idx in tqdm(range(dataset_k_fold.nb_folds), total=dataset_k_fold.nb_folds):
    global_step = 0
    for epoch in tqdm(range(config.nb_epochs), total=config.nb_epochs):
        epoch_loss = 0
        train_loader = dataset_k_fold.train_dataloader(idx)
        val_loader = dataset_k_fold.val_dataloader(idx)

        # train
        for batch, (data, label) in enumerate(train_loader):
            output = model(data).squeeze()

            loss = loss_function(output, label.float())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log train loss, train accuracy
            accuracy = accuracy_function(output, label)

            logger.log(
                {
                    f"fold_{idx}/train/loss": loss.item(),
                    f"fold_{idx}/train/acc": accuracy.item(),
                    "custom_step": global_step,
                }
            )

            global_step += 1

        # validation
        if epoch % 1 == 0:
            model.eval()
            for batch, (data, label) in enumerate(val_loader):
                with torch.no_grad():
                    output = model(data).squeeze()
                    loss = loss_function(output, label.float())

                # log train loss, train accuracy
                accuracy = accuracy_function(output, label)

                logger.log(
                    {
                        f"fold_{idx}/val/loss": loss.item(),
                        f"fold_{idx}/val/acc": accuracy.item(),
                        "custom_step": global_step,
                    }
                )

            model.train()

logger.finish()

print("Finished")
