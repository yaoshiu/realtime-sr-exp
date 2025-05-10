import glob
from typing import Generic, TypeVar
import torch
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import VideoDecoder


class VideoFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        mode="train",
        frame_step=1,
        device: str = "cuda",
        split=0.8,
    ):
        lr_paths = sorted(glob.glob(f"{lr_dir}/*"))
        hr_paths = sorted(glob.glob(f"{hr_dir}/*"))

        assert len(lr_paths) == len(hr_paths), "Mismatch between LR and HR images."

        self.mode = mode

        self.samples = []
        self.decoders = []
        for vid_idx, path in enumerate(lr_paths):
            decoder = VideoDecoder(path, device=device)
            self.decoders.append(decoder)
            n = decoder.metadata.num_frames

            tr = split
            if mode == "train":
                start, end = 0, int(n * tr)
            else:
                start, end = int(n * tr), n

            for f in range(start, end, frame_step):
                self.samples.append((vid_idx, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vid_idx, frm_idx = self.samples[idx]

        lr = self._grab_frame(vid_idx, frm_idx)
        hr = self._grab_frame(vid_idx, frm_idx)

        return lr, hr

    def _grab_frame(self, vid_idx: int, frm_idx: int) -> torch.Tensor:
        decoder = self.decoders[vid_idx]

        return decoder[frm_idx]


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
