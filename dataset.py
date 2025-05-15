import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

from utils import image_to_tensor


@pipeline_def
def image_pipeline(
    image_paths: list[str],
    crop_size: int,
    mode: str = "train",
):
    pngs, _ = fn.readers.file(name="Reader", files=image_paths)

    if mode == "train":
        pos_x = fn.random.uniform(range=(0.0, 1.0))
        pos_y = fn.random.uniform(range=(0.0, 1.0))
    else:
        pos_x = 0.5
        pos_y = 0.5

    images = fn.decoders.image_crop(
        pngs,
        crop=(crop_size, crop_size),
        crop_pos_x=pos_x,
        crop_pos_y=pos_y,
        device="mixed",
    )

    return images


class ImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        image_paths: list[str],
        crop_size: int,
        mode: str = "train",
    ):
        super().__init__()
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.mode = mode

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        hr = cv2.imread(image_path)

        h, w, _ = hr.shape

        if self.mode == "train":
            # Randomly crop the image
            x = np.random.randint(0, w - self.crop_size)
            y = np.random.randint(0, h - self.crop_size)
            hr = hr[y : y + self.crop_size, x : x + self.crop_size]

        else:
            # Center crop the image
            x = (w - self.crop_size) // 2
            y = (h - self.crop_size) // 2
            hr = hr[y : y + self.crop_size, x : x + self.crop_size]

        hr = image_to_tensor(hr)

        return hr

    def __len__(self) -> int:
        return len(self.image_paths)


class CUDAPrefetcher:
    def __init__(
        self,
        dataloader: DataLoader[torch.Tensor],
        device: torch.device | str,
    ):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):  # type: ignore
            if torch.is_tensor(self.batch_data):
                self.batch_data = self.batch_data.to(self.device, non_blocking=True)

    def next(self) -> torch.Tensor | None:
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
