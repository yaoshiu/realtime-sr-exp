from enum import Enum
from typing import Optional
import cv2
import torch
import torch.nn as nn
import model
import numpy as np

import torch.nn.functional as F
import torchvision.transforms.functional as TF


def choice_device(device_type: str):
    return torch.device(device_type, 0)


def build_model(model_arch_name: str, device: torch.device | None = None) -> nn.Module:
    sr_model = model.__dict__[model_arch_name](
        in_channels=1, out_channels=1, channels=64
    )
    if device:
        sr_model = sr_model.to(device=device)

    return sr_model


@torch.jit.script
def bgr_to_y_torch(bgr_tensor: torch.Tensor):
    weight = torch.tensor([[24.966], [128.553], [65.481]]).to(bgr_tensor)

    bias = 16.0

    y_tensor = (
        torch.matmul(bgr_tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    )

    return y_tensor / 255.0


@torch.jit.script
def bgr_to_ycbcr_torch(bgr_tensor: torch.Tensor):
    weight = torch.tensor(
        [
            [24.966, 112.0, -18.214],
            [128.553, -74.203, -93.786],
            [65.481, -37.797, 112.0],
        ],
    ).to(bgr_tensor)

    bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(bgr_tensor)

    ycbcr_tensor = (
        torch.matmul(bgr_tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    )

    return ycbcr_tensor / 255.0


@torch.jit.script
def rgb_to_y_torch(rgb_tensor: torch.Tensor):
    weight = torch.tensor([[65.481], [128.553], [24.966]]).to(rgb_tensor)

    bias = 16.0

    y_tensor = (
        torch.matmul(rgb_tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    )

    return y_tensor / 255.0


@torch.jit.script
def rgb_to_ycbcr_torch(rgb_tensor: torch.Tensor):
    weight = torch.tensor(
        [
            [65.481, -37.797, 112.0],
            [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214],
        ],
    ).to(rgb_tensor)

    bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(rgb_tensor)

    ycbcr_tensor = (
        torch.matmul(rgb_tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    )

    return ycbcr_tensor / 255.0


@torch.jit.script
def ycbcr_to_bgr_torch(ycbcr_tensor: torch.Tensor):
    weight = (
        torch.tensor(
            [
                [298.082, 298.082, 298.082],
                [516.412, -100.291, 0.0],
                [0.0, -208.120, 408.583],
            ],
        )
    ).to(ycbcr_tensor) / 256.0

    bias = (
        torch.tensor([-276.836, 135.576, -222.921]).view(1, 3, 1, 1).to(ycbcr_tensor)
        / 255.0
    )

    bgr_tensor = (
        torch.matmul(ycbcr_tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2)
        + bias
    )

    return bgr_tensor


@torch.jit.script
def ycbcr_to_rgb_torch(ycbcr_tensor: torch.Tensor):
    weight = (
        torch.tensor(
            [
                [298.082, 298.082, 298.082],
                [0.0, -100.291, 516.412],
                [408.583, -208.120, 0.0],
            ],
        )
    ).to(ycbcr_tensor) / 256.0

    bias = (
        torch.tensor([-222.921, 135.576, -276.836]).view(1, 3, 1, 1).to(ycbcr_tensor)
        / 255.0
    )

    rgb_tensor = (
        torch.matmul(ycbcr_tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2)
        + bias
    )

    return rgb_tensor


@torch.jit.script
def bgr_to_rgb_torch(bgr_tensor: torch.Tensor):
    rgb_tensor = torch.flip(bgr_tensor, [1])

    return rgb_tensor


def image_to_tensor(image: np.ndarray, device: torch.device | str = "cpu", half=False):
    tensor = torch.from_numpy(image).to(device).permute(2, 0, 1).float() / 255.0

    tensor = tensor.unsqueeze_(0)

    if half:
        tensor = tensor.half()

    return tensor


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def _check_tensor_shape(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]

    """
    # Check if tensor scales are consistent
    assert (
        raw_tensor.shape == dst_tensor.shape
    ), f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def _psnr_torch(
    raw_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    crop_border: int,
    only_test_y_channel: bool,
) -> torch.Tensor:
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics

    """
    # Check if two tensor scales are similar
    _check_tensor_shape(raw_tensor, dst_tensor)

    # crop border pixels
    if crop_border > 0:
        raw_tensor = raw_tensor[
            :, :, crop_border:-crop_border, crop_border:-crop_border
        ]
        dst_tensor = dst_tensor[
            :, :, crop_border:-crop_border, crop_border:-crop_border
        ]

    # Convert RGB tensor data to YCbCr tensor, and extract only Y channel data
    if only_test_y_channel:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, only_use_y_channel=True)

    # Convert data type to torch.float64 bit
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)

    mse_value = torch.mean(
        (raw_tensor * 255.0 - dst_tensor * 255.0) ** 2 + 1e-8, dim=[1, 2, 3]
    )
    psnr_metrics = 10 * torch.log10_(255.0**2 / mse_value)

    return psnr_metrics


class PSNR(nn.Module):
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Attributes:
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics

    """

    def __init__(self, crop_border: int, only_test_y_channel: bool) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel

    def forward(
        self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor
    ) -> torch.Tensor:
        psnr_metrics = _psnr_torch(
            raw_tensor, dst_tensor, self.crop_border, self.only_test_y_channel
        )

        return psnr_metrics


def _ssim_torch(
    raw_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    window_size: int,
    gaussian_kernel_window: np.ndarray,
) -> torch.Tensor:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 255]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 255]
        window_size (int): Gaussian filter size
        gaussian_kernel_window (np.ndarray): Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    gkw_tensor = torch.from_numpy(gaussian_kernel_window).view(
        1, 1, window_size, window_size
    )
    gkw_tensor = gkw_tensor.expand(raw_tensor.size(1), 1, window_size, window_size)
    gkw_tensor = gkw_tensor.to(device=raw_tensor.device, dtype=raw_tensor.dtype)

    raw_mean = F.conv2d(
        raw_tensor,
        gkw_tensor,
        stride=(1, 1),
        padding=(0, 0),
        groups=raw_tensor.shape[1],
    )
    dst_mean = F.conv2d(
        dst_tensor,
        gkw_tensor,
        stride=(1, 1),
        padding=(0, 0),
        groups=dst_tensor.shape[1],
    )
    raw_mean_square = raw_mean**2
    dst_mean_square = dst_mean**2
    raw_dst_mean = raw_mean * dst_mean
    raw_variance = (
        F.conv2d(
            raw_tensor * raw_tensor,
            gkw_tensor,
            stride=(1, 1),
            padding=(0, 0),
            groups=raw_tensor.shape[1],
        )
        - raw_mean_square
    )
    dst_variance = (
        F.conv2d(
            dst_tensor * dst_tensor,
            gkw_tensor,
            stride=(1, 1),
            padding=(0, 0),
            groups=raw_tensor.shape[1],
        )
        - dst_mean_square
    )
    raw_dst_covariance = (
        F.conv2d(
            raw_tensor * dst_tensor,
            gkw_tensor,
            stride=1,
            padding=(0, 0),
            groups=raw_tensor.shape[1],
        )
        - raw_dst_mean
    )

    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (
        raw_variance + dst_variance + c2
    )

    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = torch.mean(ssim_metrics, [1, 2, 3]).float()

    return ssim_metrics


def _ssim_single_torch(
    raw_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    crop_border: int,
    only_test_y_channel: bool,
    window_size: int,
    gaussian_kernel_window: np.ndarray,
) -> torch.Tensor:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_kernel_window (ndarray): Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """
    # Check if two tensor scales are similar
    _check_tensor_shape(raw_tensor, dst_tensor)

    # crop border pixels
    if crop_border > 0:
        raw_tensor = raw_tensor[
            :, :, crop_border:-crop_border, crop_border:-crop_border
        ]
        dst_tensor = dst_tensor[
            :, :, crop_border:-crop_border, crop_border:-crop_border
        ]

    # Convert RGB tensor data to YCbCr tensor, and extract only Y channel data
    if only_test_y_channel:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, only_use_y_channel=True)

    # Convert data type to torch.float64 bit
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)

    ssim_metrics = _ssim_torch(
        raw_tensor * 255.0, dst_tensor * 255.0, window_size, gaussian_kernel_window
    )

    return ssim_metrics


class SSIM(nn.Module):
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        crop_border (int): crop border a few pixels
        only_only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_sigma (float): sigma parameter in Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """

    def __init__(
        self,
        crop_border: int,
        only_only_test_y_channel: bool,
        window_size: int = 11,
        gaussian_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_only_test_y_channel
        self.window_size = window_size

        gaussian_kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
        self.gaussian_kernel_window = np.outer(
            gaussian_kernel, gaussian_kernel.transpose()
        )

    def forward(
        self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor
    ) -> torch.Tensor:
        ssim_metrics = _ssim_single_torch(
            raw_tensor,
            dst_tensor,
            self.crop_border,
            self.only_test_y_channel,
            self.window_size,
            self.gaussian_kernel_window,
        )

        return ssim_metrics
