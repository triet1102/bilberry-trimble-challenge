from torch.utils.data import Dataset
from PIL import Image
from src.utils.helper_functions import get_image_transforms, get_image_augmentations


class FieldRoadDataset(Dataset):
    """Field Road Dataset"""

    def __init__(
        self,
        files: list[str],
        labels: list[int],
        class_names_to_idx: dict[str, int],
        augmentation: bool,
    ):
        """
        Create a FieldRoadDataset object
        """
        self.image_transforms = (
            get_image_transforms() if not augmentation else get_image_augmentations()
        )
        self.class_names_to_idx = class_names_to_idx
        self.files = files
        self.labels = labels

    def __len__(self) -> int:
        """Return the total number of images in the dataset"""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        """Get an item of the dataset"""
        file_name = self.files[idx]

        # use PIL to read image
        image = Image.open(file_name)
        image = self.image_transforms(image)

        return image, self.class_names_to_idx[self.labels[idx]]
