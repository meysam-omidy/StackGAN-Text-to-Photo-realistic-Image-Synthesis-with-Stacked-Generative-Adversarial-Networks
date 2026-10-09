"""Dataset and network building blocks used by both StackGAN stages."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "CUB_200_2011" / "images"
TEXT_ROOT = PROJECT_ROOT / "birds" / "text_c10"
ANNOTATION_ROOT = PROJECT_ROOT / "CUB_200_2011"


def load_bounding_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Return CUB bounding boxes keyed by their image path."""
    image_paths = pd.read_csv(
        ANNOTATION_ROOT / "images.txt",
        sep=r"\s+",
        names=["image_id", "path"],
    )
    boxes = pd.read_csv(
        ANNOTATION_ROOT / "bounding_boxes.txt",
        sep=r"\s+",
        names=["image_id", "x", "y", "width", "height"],
    )
    annotations = image_paths.merge(boxes, on="image_id", validate="one_to_one")
    if len(annotations) != len(image_paths):
        raise ValueError("CUB image paths and bounding boxes do not form a one-to-one mapping")
    return {
        row.path: (row.x, row.y, row.width, row.height)
        for row in annotations.itertuples(index=False)
    }


class BirdTextDataset(Dataset):
    """CUB image/caption-embedding pairs from the original StackGAN split."""

    def __init__(self, split: str, image_size: int, training: bool, bbox_padding: float = 0.35) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        metadata_dir = PROJECT_ROOT / "birds" / split
        self.filenames = pd.read_pickle(metadata_dir / "filenames.pickle")
        embeddings_path = metadata_dir / "char-CNN-RNN-embeddings.pickle"
        embeddings = np.asarray(pd.read_pickle(embeddings_path), dtype=np.float32)
        if embeddings.ndim != 3 or embeddings.shape[1:] != (10, 1024):
            raise ValueError(f"Unexpected embedding shape {embeddings.shape} in {embeddings_path}")
        if len(self.filenames) != len(embeddings):
            raise ValueError("The filenames and caption embeddings have different lengths")
        self.embeddings = torch.from_numpy(embeddings)
        self.image_paths = [IMAGE_ROOT / f"{name}.jpg" for name in self.filenames]
        self.caption_paths = [TEXT_ROOT / f"{name}.txt" for name in self.filenames]
        if bbox_padding < 0:
            raise ValueError("bbox_padding must be non-negative")
        bounding_boxes = load_bounding_boxes()
        self.bounding_boxes = []
        for filename in self.filenames:
            box = bounding_boxes.get(f"{filename}.jpg")
            if box is None:
                raise KeyError(f"No CUB bounding box found for {filename}")
            self.bounding_boxes.append(box)
        self.bbox_padding = bbox_padding
        missing = [str(path) for path in self.image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} CUB images; first: {missing[0]}")

        resize_size = int(round(image_size * 76 / 64))
        transforms_list = [transforms.Resize(resize_size)]
        if training:
            transforms_list += [transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip()]
        else:
            transforms_list += [transforms.CenterCrop(image_size)]
        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(transforms_list)
        self.training = training

    def __len__(self) -> int:
        return len(self.filenames)

    def _crop_bird_with_context(self, image: Image.Image, index: int) -> Image.Image:
        """Crop a padded square around the bird, retaining natural background context."""
        x, y, width, height = self.bounding_boxes[index]
        center_x, center_y = x + width / 2, y + height / 2
        side = max(width, height) * (1 + 2 * self.bbox_padding)
        # Keep the crop inside the source image. Shifting is preferable to black padding.
        side = min(side, image.width, image.height)
        left = min(max(center_x - side / 2, 0), image.width - side)
        top = min(max(center_y - side / 2, 0), image.height - side)
        right, bottom = left + side, top + side
        return image.crop((round(left), round(top), round(right), round(bottom)))

    def __getitem__(self, index: int):
        with Image.open(self.image_paths[index]) as image:
            image = self._crop_bird_with_context(image.convert("RGB"), index)
            image = self.transform(image)
        caption_index = torch.randint(10, ()).item() if self.training else 0
        return image, self.embeddings[index, caption_index], index

    def caption(self, index: int) -> list[str]:
        """Return raw captions when the corresponding text files are present."""
        path = self.caption_paths[index]
        return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def get_data(batch_size: int, image_size: int, split: str = "train", training: bool = True, num_workers: int = 0, bbox_padding: float = 0.35) -> DataLoader:
    """Return a paired loader: images and caption embeddings cannot drift apart."""
    dataset = BirdTextDataset(split=split, image_size=image_size, training=training, bbox_padding=bbox_padding)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def upblock(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def downblock(in_channels: int, out_channels: int, with_batchnorm: bool = True, spectral: bool = False) -> nn.Sequential:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
    if spectral:
        conv = nn.utils.spectral_norm(conv)
    layers = [conv]
    if with_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Residual_Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layer(x) + x)


class C_Net(nn.Module):
    """Conditioning augmentation and its Gaussian parameters."""

    def __init__(self, embedding_size: int, condition_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(embedding_size, 2 * condition_size)

    def forward(self, embedding: torch.Tensor):
        mu, logvar = self.layer(embedding).chunk(2, dim=1)
        condition = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return condition, mu, logvar
