import time
import argparse
from typing import Callable

import torch
import numpy as np
import cv2
from torch import nn
from torch.nn import functional as F

import cudacanvas

import model


model_names = sorted(
    name
    for name in model.__dict__
    if name.islower() and not name.startswith("__") and callable(model.__dict__[name])
)

torch.backends.cudnn.benchmark = True


def choice_device(device_type: str) -> torch.device:
    return torch.device("cuda", 0) if device_type == "cuda" else torch.device("cpu")


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    sr_model = model.__dict__[model_arch_name](
        in_channels=1, out_channels=1, channels=64
    )
    sr_model = sr_model.to(device=device)

    return sr_model


@torch.jit.script
def bgr_to_ycbcr_tensor(bgr_tensor: torch.Tensor):
    BGR2YCBCR_WEIGHT = torch.tensor(
        [
            [24.966, 112.0, -18.214],
            [128.553, -74.203, -93.786],
            [65.481, -37.797, 112.0],
        ],
        device=torch.device("cuda"),
        dtype=torch.half,
    )

    BGR2YCBCR_BIAS = torch.tensor(
        [16, 128, 128], device=torch.device("cuda"), dtype=torch.half
    ).view(1, 3, 1, 1)

    ycbcr_tensor = (
        # B x C x H x W -> B x H x W x C
        torch.matmul(bgr_tensor.permute(0, 2, 3, 1), BGR2YCBCR_WEIGHT).permute(
            0, 3, 1, 2
        )
        + BGR2YCBCR_BIAS
    )

    return ycbcr_tensor / 255.0


@torch.jit.script
def ycbcr_to_bgr_tensor(ycbcr_tensor: torch.Tensor):
    weight = torch.tensor(
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0.00791071, -0.00153632, 0.0],
            [0.0, -0.00318811, 0.00625893],
        ],
        device=torch.device("cuda"),
        dtype=torch.half,
    )

    bias = torch.tensor(
        [-276.836, 135.576, -222.921], device=torch.device("cuda"), dtype=torch.half
    ).view(1, 3, 1, 1)

    ycbcr_tensor *= 255.0

    bgr_tensor = (
        torch.matmul(ycbcr_tensor.permute(0, 2, 3, 1), weight) * 255.0
    ).permute(0, 3, 1, 2) + bias

    return bgr_tensor / 255.0


@torch.jit.script
def ycbcr_to_rgb_tensor(ycbcr_tensor: torch.Tensor):
    weight = torch.tensor(
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0.0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0.0],
        ],
        device=torch.device("cuda"),
        dtype=torch.half,
    )

    bias = torch.tensor(
        [-222.921, 135.576, -276.836], device=torch.device("cuda"), dtype=torch.half
    ).view(1, 3, 1, 1)

    ycbcr_tensor *= 255.0

    rgb_tensor = (
        torch.matmul(ycbcr_tensor.permute(0, 2, 3, 1), weight) * 255.0
    ).permute(0, 3, 1, 2) + bias

    return rgb_tensor / 255.0


def image_to_tensor(image: np.ndarray, device: torch.device):
    tensor = torch.tensor(image, dtype=torch.half, device=device) / 255.0

    # H x W x C -> C x H x W
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor


@torch.jit.script
def preprocess(input: torch.Tensor, upscale_factor: float):
    ycbcr_tensor = bgr_to_ycbcr_tensor(input)

    lr_y_tensor, bic_cb_tensor, bic_cr_tensor = torch.split(ycbcr_tensor, 1, dim=1)

    bic_cb_tensor = F.interpolate(
        bic_cb_tensor,
        scale_factor=upscale_factor,
        mode="bilinear",
        align_corners=False,
    )

    bic_cr_tensor = F.interpolate(
        bic_cr_tensor,
        scale_factor=upscale_factor,
        mode="bilinear",
        align_corners=False,
    )

    return lr_y_tensor, bic_cb_tensor, bic_cr_tensor


@torch.jit.script
def postprocess(
    sr_y_tensor: torch.Tensor, bic_cb_tensor: torch.Tensor, bic_cr_tensor: torch.Tensor
):
    sr_ycbcr_tensor = torch.cat([sr_y_tensor, bic_cb_tensor, bic_cr_tensor], dim=1)

    return ycbcr_to_rgb_tensor(sr_ycbcr_tensor).clamp_(0.0, 1.0)


def process_frame(
    sr_model: Callable[[torch.Tensor], torch.Tensor],
    frame: np.ndarray,
    upscale_factor: float,
    device: torch.device,
):
    tensor = image_to_tensor(frame, device)

    with torch.no_grad():
        lr_y_tensor, bic_cb_tensor, bic_cr_tensor = preprocess(tensor, upscale_factor)

        sr_y_tensor = sr_model(lr_y_tensor)

        return postprocess(sr_y_tensor, bic_cb_tensor, bic_cr_tensor)


def main(args):
    device = choice_device("cuda")

    sr_model = build_model(args.model_arch_name, device)

    checkpoint = torch.load(
        args.model_weights_path, map_location=lambda storage, loc: storage
    )
    sr_model.load_state_dict(checkpoint["state_dict"], strict=False)

    if not isinstance(sr_model, nn.Module):
        print("Error: Failed to load model.")
        return

    sr_model.half()
    sr_model.eval()
    sr_model = torch.jit.script(sr_model)

    cap = cv2.VideoCapture(args.input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    init = time.perf_counter()
    last_time = 0
    frames = 0

    max_delay = args.max_delay

    upscale_factor = args.upscale_factor

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        src_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        if time.perf_counter() - init - src_time / 1000 > max_delay:
            continue

        sr_frame = process_frame(sr_model, frame, upscale_factor, device)

        frames += 1

        current = time.perf_counter()
        if current - last_time >= 1:
            print(f"FPS: {frames / (current - last_time):.2f}")
            print(f"Delay: {current - init - src_time / 1000:.4f}")
            frames = 0
            last_time = current

        cudacanvas.im_show(sr_frame.squeeze(0))

        if cudacanvas.should_close():
            break

    cap.release()
    cudacanvas.clean_up()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-Resolution on video stream.")
    parser.add_argument("--max_delay", type=float, default=0.02)
    parser.add_argument("--model_arch_name", type=str, default="espcn_x2")
    parser.add_argument("--upscale_factor", type=int, default=2)
    parser.add_argument("--input_video_path", type=str, default="./figure/video.mp4")
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./results/pretrained_models/ESPCN_x2-T91-da809cd7.pth.tar",
    )
    args = parser.parse_args()
    main(args)
