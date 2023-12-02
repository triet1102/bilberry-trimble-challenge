import torch
import timm


class ClassificationModel(torch.nn.Module):
    """Classification Model"""

    def __init__(
        self,
        num_classes: int,
        backbone_ckpt: str = "convnextv2_tiny.fcmae",
    ):
        """Create a ClassificationModel object

        Args:
            num_classes: Number of classes.
        """
        super().__init__()
        self.num_classes = num_classes

        # load the backbone
        self.backbone = timm.create_model(backbone_ckpt, pretrained=True)
        self.embedding_dim = 768
        # print(self.backbone)
        self.fc = torch.nn.Linear(self.embedding_dimn, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.backbone(x)
        x = self.fc(x)
        return x


# from urllib.request import urlopen
# from PIL import Image
# import timm

# img = Image.open(urlopen(
#     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# ))

# model = timm.create_model(
#     'convnextv2_tiny.fcmae',
#     pretrained=True,
#     num_classes=0,  # remove classifier nn.Linear
# )
# model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# # or equivalently (without needing to set num_classes=0)

# output = model.forward_features(transforms(img).unsqueeze(0))
# # output is unpooled, a (1, 768, 7, 7) shaped tensor

# output = model.forward_head(output, pre_logits=True)
# # output is a (1, num_features) shaped tensor
