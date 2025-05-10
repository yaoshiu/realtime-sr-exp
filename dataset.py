import glob
from math import e
from typing import Generic, TypeVar
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from utils import image_to_tensor


class VideoFrameDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        upscale_factor: float,
        frame_step=1,
        cache=False,
        mode="train",
        split=0.8,
    ):
        self.lr_paths = sorted(glob.glob(f"{lr_dir}/*"))
        self.hr_paths = sorted(glob.glob(f"{hr_dir}/*"))

        assert len(self.lr_paths) == len(
            self.hr_paths
        ), "Mismatch between LR and HR images."

        self.upscale_factor = upscale_factor
        self.frame_step = frame_step
        self.cache = cache
        self.mode = mode

        self.samples = []
        self.frame_ranges = []
        for vid_idx, path in enumerate(self.lr_paths):
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            tr = split
            if mode == "train":
                start, end = 0, int(n * tr)
            else:
                start, end = int(n * tr), n

            for f in range(start, end, frame_step):
                self.samples.append((vid_idx, f))

        if cache:
            self.lr_cache, self.hr_cache = {}, {}
            for vid_idx, (lp, hp) in enumerate(zip(self.lr_paths, self.hr_paths)):
                self.lr_cache[vid_idx] = self._decode_all(lp)
                self.hr_cache[vid_idx] = self._decode_all(hp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid_idx, frm_idx = self.samples[idx]

        lr = self._grab_frame(self.lr_paths[vid_idx], vid_idx, frm_idx, is_lr=True)
        hr = self._grab_frame(self.hr_paths[vid_idx], vid_idx, frm_idx, is_lr=False)

        lr = image_to_tensor(lr)
        hr = image_to_tensor(hr)

        lr = lr.squeeze_(0)
        hr = hr.squeeze_(0)

        return {"lr": lr, "hr": hr}

    def _grab_frame(
        self, path: str, vid_idx: int, frm_idx: int, *, is_lr=True
    ) -> np.ndarray:
        if self.cache:
            buf = self.lr_cache if is_lr else self.hr_cache
            return buf[vid_idx][frm_idx]

        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
        _, frame = cap.read()
        cap.release()

        return frame

    def _decode_all(self, path: str) -> list[torch.Tensor]:
        cap, frames = cv2.VideoCapture(path), []
        ret, f = cap.read()
        while ret:
            frames.append(f)
            ret, f = cap.read()
        cap.release()
        return frames


T = TypeVar("T")


class CUDAPrefetcher(Generic[T]):
    def __init__(self, dataloader: DataLoader[T], device: torch.device):
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
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(
                        self.device, non_blocking=True
                    )

    def next(self) -> T | None:
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
