import time
import argparse
from typing import Callable

import torch
import numpy as np
import cv2
from torch import nn
from torch import amp
from torch.nn import functional as F

import cudacanvas

from utils import *


torch.backends.cudnn.benchmark = True


def espcn_x2(
    sr_model: Callable[[torch.Tensor], torch.Tensor],
    frame: np.ndarray,
    upscale_factor: float,
    device: torch.device,
):
    tensor = image_to_tensor(frame, device)

    with torch.no_grad():
        ycbcr_tensor = bgr_to_ycbcr_torch(tensor)

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

        sr_y_tensor = sr_model(lr_y_tensor)

        sr_y_tensor = F.interpolate(
            sr_y_tensor,
            scale_factor=upscale_factor / 2,
            mode="bilinear",
            align_corners=False,
        )

        sr_ycbcr_tensor = torch.cat([sr_y_tensor, bic_cb_tensor, bic_cr_tensor], dim=1)

        result = ycbcr_to_rgb_torch(sr_ycbcr_tensor).clamp_(0.0, 1.0)

        return result


def fsrcnn_x2(
    sr_model: Callable[[torch.Tensor], torch.Tensor],
    frame: np.ndarray,
    upscale_factor: float,
    device: torch.device,
):
    tensor = image_to_tensor(frame, device)

    with torch.no_grad():
        ycbcr_tensor = bgr_to_ycbcr_torch(tensor)

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

        sr_y_tensor = sr_model(lr_y_tensor)

        sr_y_tensor = F.interpolate(
            sr_y_tensor,
            scale_factor=upscale_factor / 2,
            mode="bilinear",
            align_corners=False,
        )

        sr_ycbcr_tensor = torch.cat([sr_y_tensor, bic_cb_tensor, bic_cr_tensor], dim=1)

        result = ycbcr_to_rgb_torch(sr_ycbcr_tensor).clamp_(0.0, 1.0)

        return result


def rt4ksr_rep(
    sr_model: Callable[[torch.Tensor], torch.Tensor],
    frame: np.ndarray,
    upscale_factor: float,
    device: torch.device,
):
    tensor = image_to_tensor(frame, device)

    with torch.no_grad():
        rgb_tensor = bgr_to_rgb_torch(tensor)

        result = sr_model(rgb_tensor).clamp_(0.0, 1.0)

        result = F.interpolate(
            result,
            scale_factor=upscale_factor / 2,
            mode="bilinear",
            align_corners=False,
        )

        return result


def main(args):
    device = choice_device(args.device)

    sr_model = build_model(args.model_arch_name, device)

    checkpoint = torch.load(
        args.model_weights_path, map_location=lambda storage, _: storage
    )
    sr_model.load_state_dict(checkpoint["state_dict"], strict=True)

    if not isinstance(sr_model, nn.Module):
        print("Error: Failed to load model.")
        return

    sr_model.eval()

    cap = cv2.VideoCapture(args.input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    max_delay = args.max_delay

    upscale_factor = args.upscale_factor

    init = time.perf_counter()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Completed.")
            break

        src_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        delay = time.perf_counter() - init - src_time / 1000
        if delay > max_delay:
            continue

        if delay < 0:
            time.sleep(-delay)

        with amp.autocast_mode.autocast(args.device):
            sr_frame = globals()[args.model_arch_name](
                sr_model, frame, upscale_factor, device
            )

        frames += 1

        cudacanvas.im_show(sr_frame.squeeze_(0))

        if cudacanvas.should_close():
            break

    avg_frames = frames / (time.perf_counter() - init)
    print(f"Average FPS: {avg_frames:.2f}")

    hr = cv2.imread(args.image_hr)
    lr = cv2.imread(args.image_lr)

    sr = globals()[args.model_arch_name](sr_model, lr, upscale_factor, device)

    hr = image_to_tensor(hr, device)
    hr = bgr_to_rgb_torch(hr).clamp_(0.0, 1.0)

    psnr_model = PSNR(0, False)
    ssim_model = SSIM(0, False)

    psnr_metrics = psnr_model(sr, hr)
    ssim_metrics = ssim_model(sr, hr)

    psnr = psnr_metrics.item()
    ssim = ssim_metrics.item()

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

    cap.release()
    cudacanvas.clean_up()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-Resolution on video stream.")
    parser.add_argument("--max_delay", type=float, default=0.02)
    parser.add_argument("--model_arch_name", type=str, default="fsrcnn_x2")
    parser.add_argument("--upscale_factor", type=float, default=1.5)
    parser.add_argument(
        "--input_video_path", type=str, default="./data/validate/video.mov"
    )
    parser.add_argument("--image_hr", type=str, default="./data/validate/HR.png")
    parser.add_argument("--image_lr", type=str, default="./data/validate/LR.png")
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./results/pretrained_models/fsrcnn_x2-T91-f791f07f.pth.tar",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
