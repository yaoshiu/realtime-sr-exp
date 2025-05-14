from typing import Callable
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

from utils import image_to_tensor


class ImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        image_paths: list[str],
        upscale_factor: float,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.upscale_factor = upscale_factor

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        hr = cv2.imread(image_path)

        hr = image_to_tensor(hr)

        return hr

    def __len__(self) -> int:
        return len(self.image_paths)


class CUDAPrefetcher:
    def __init__(
        self,
        dataloader: DataLoader[torch.Tensor],
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str,
    ):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device
        self.preprocess = preprocess

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
        if torch.is_tensor(batch_data):
            batch_data = self.preprocess(batch_data)
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
