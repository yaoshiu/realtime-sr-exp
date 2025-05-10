import glob
from typing import Generic, TypeVar
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


class VideoFrameDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        mode="train",
        frame_step=1,
        device: torch.device | str = "cuda",
        split=0.8,
    ):
        lr_paths = sorted(glob.glob(f"{lr_dir}/*"))
        hr_paths = sorted(glob.glob(f"{hr_dir}/*"))

        assert len(lr_paths) == len(hr_paths), "Mismatch between LR and HR images."
        assert len(lr_paths) > 0, "No images found in the specified directories."

        self.mode = mode
        self.samples = []
        self.decoders: list[tuple[cv2.VideoCapture, cv2.VideoCapture]] = []

        for vid_idx, (lr_path, hr_path) in enumerate(zip(lr_paths, hr_paths)):
            lr_decoder = cv2.VideoCapture(lr_path)
            hr_decoder = cv2.VideoCapture(hr_path)

            self.decoders.append((lr_decoder, hr_decoder))

            lr_frames = int(lr_decoder.get(cv2.CAP_PROP_FRAME_COUNT))
            hr_frames = int(hr_decoder.get(cv2.CAP_PROP_FRAME_COUNT))
            n = min(lr_frames, hr_frames)

            if mode == "train":
                start, end = 0, int(n * split)
            else:
                start, end = int(n * split), n

            for f in range(start, end, frame_step):
                self.samples.append((vid_idx, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        vid_idx, frm_idx = self.samples[idx]

        return self._grab_frame(vid_idx, frm_idx)

    def _grab_frame(self, vid_idx: int, frm_idx: int) -> tuple[np.ndarray, np.ndarray]:
        lr_decoder, hr_decoder = self.decoders[vid_idx]

        lr_decoder.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
        ret_lr, lr = lr_decoder.read()
        if not ret_lr:
            raise RuntimeError(f"Failed to read frame {frm_idx} from video {vid_idx}")

        hr_decoder.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
        ret_hr, hr = hr_decoder.read()
        if not ret_hr:
            raise RuntimeError(f"Failed to read frame {frm_idx} from video {vid_idx}")

        return lr, hr


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
