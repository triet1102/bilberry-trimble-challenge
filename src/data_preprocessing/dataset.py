from torch.utils.data import Dataset
from collections.abc import Callable
from PIL import Image


class FieldRoadDataset(Dataset):
    """Field Road Dataset"""

    def __init__(
        self,
        files: list[str],
        labels: list[int],
        class_names_to_idx: dict[str, int],
        transforms: Callable | None = None,
    ):
        """Create a FieldRoadDataset object

        Args:
            root_dir: Path to the data directory.
            transforms: Optional transforms to be applied on a sample.
        """
        self.transforms = transforms

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

        # mandantory transform for imagenet
        # image = self.imagenet_transform(image=image)["image"]

        # apply optional transforms if provided (e.g Resize, Rotate, Translate, etc.)
        if self.transforms:
            image = self.transforms(image)

        return image, self.class_names_to_idx[self.labels[idx]]
