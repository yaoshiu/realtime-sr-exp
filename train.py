from functools import partial
from glob import glob
import os
import shutil
import time

import argparse
from sklearn.model_selection import train_test_split
import torch
from torch import amp, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import DataParallel
from dataset import ImageDataset, CUDAPrefetcher
from utils import *
from torch.profiler import profile, ProfilerActivity


def load_dataset(
    image_dir: str,
    *,
    device: torch.device | str,
    upscale_factor: float,
    split=0.8,
    batch_size=64,
    num_workers=12,
):
    tr_paths, te_paths = train_test_split(
        glob(os.path.join(image_dir, "*.png")),
        test_size=1 - split,
        random_state=42,
        shuffle=True,
        stratify=None,
    )

    tr_set = ImageDataset(
        tr_paths,
        upscale_factor=upscale_factor,
        crop_size=int(256 * upscale_factor),
        mode="train",
    )

    te_set = ImageDataset(
        te_paths,
        upscale_factor=upscale_factor,
        crop_size=int(256 * upscale_factor),
        mode="test",
    )

    tr_loader = DataLoader(
        tr_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    te_loader = DataLoader(
        te_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )

    tr_prefetcher = CUDAPrefetcher(
        tr_loader,
        device=device,
    )
    te_prefetcher = CUDAPrefetcher(
        te_loader,
        device=device,
    )

    return tr_prefetcher, te_prefetcher


def train(
    model: torch.nn.Module,
    train_loader: CUDAPrefetcher,
    criterion: torch.nn.MSELoss,
    optimizer: optim.SGD,
    upscale_factor: float,
    epoch: int,
    scaler: amp.grad_scaler.GradScaler,
    writer: SummaryWriter,
    print_freq=200,
):
    batches = len(train_loader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(
        batches,
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    model.train()

    batch_index = 0

    end = time.time()

    train_loader.reset()
    batch_data = train_loader.next()

    while batch_data is not None:
        hr = batch_data

        data_time.update(time.time() - end)

        hr = bgr_to_y_torch(hr)

        lr = F.interpolate(
            hr,
            scale_factor=1 / upscale_factor,
            mode="bilinear",
            align_corners=False,
        )

        model.zero_grad()

        with amp.autocast_mode.autocast("cuda"):
            sr = model(lr)
            if upscale_factor != 2:
                sr = F.interpolate(
                    sr,
                    scale_factor=upscale_factor / 2,
                    mode="bilinear",
                    align_corners=False,
                )
            loss = criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), lr.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % print_freq == 0:
            writer.add_scalar(
                "Train/Loss", loss.item(), batch_index + epoch * batches + 1
            )
            progress.display(batch_index + 1)

        batch_index += 1

        batch_data = train_loader.next()


def validate(
    model: torch.nn.Module,
    valid_loader: CUDAPrefetcher,
    psnr_model: PSNR,
    ssim_model: SSIM,
    epoch: int,
    upscale_factor: float,
    writer: SummaryWriter,
    mode="val",
    print_freq=200,
):
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(
        len(valid_loader),
        [batch_time, psnres, ssimes],
        prefix=f"{mode}: ",
    )

    model.eval()

    batch_index = 0

    end = time.time()

    valid_loader.reset()
    batch_data = valid_loader.next()

    with torch.no_grad():
        while batch_data is not None:
            hr = batch_data

            hr = bgr_to_ycbcr_torch(hr)

            lr = F.interpolate(
                hr,
                scale_factor=1 / upscale_factor,
                mode="bilinear",
                align_corners=False,
            )

            with amp.autocast_mode.autocast("cuda"):
                lr_y, lr_cb, lr_cr = torch.split(lr, 1, dim=1)

                sr_cb = F.interpolate(
                    lr_cb,
                    scale_factor=upscale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                sr_cr = F.interpolate(
                    lr_cr,
                    scale_factor=upscale_factor,
                    mode="bilinear",
                    align_corners=False,
                )

                sr_y = model(lr_y)

                if upscale_factor != 2:
                    sr_y = torch.nn.functional.interpolate(
                        sr_y,
                        scale_factor=upscale_factor / 2,
                        mode="bilinear",
                        align_corners=False,
                    )

                sr = torch.cat([sr_y, sr_cb, sr_cr], dim=1)

            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)

            psnres.update(psnr.mean().item(), lr.size(0))
            ssimes.update(ssim.mean().item(), lr.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % print_freq == 0:
                progress.display(batch_index + 1)

            batch_index += 1

            batch_data = valid_loader.next()

    progress.display_summary()

    writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
    writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)

    return psnres.avg, ssimes.avg


def main(args):
    start_epoch = 0

    best_psnr = 0.0
    best_ssim = 0.0

    device = choice_device(args.device)
    print(f"Using `{device}` device.")

    model = build_model(args.model_arch_name)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device=device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    train_loader, test_loader = load_dataset(
        args.image_dir, device=device, upscale_factor=args.upscale_factor
    )
    print("Load datasets successfully.")

    criterion = torch.nn.MSELoss().to(device=device)
    print("Define loss function successfully.")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.model_lr,
        momentum=args.model_momentum,
        weight_decay=args.model_weight_decay,
        nesterov=args.model_nesterov,
    )
    print("Define optimizer successfully.")

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.model_milestones, gamma=args.model_gamma
    )
    print("Define scheduler successfully.")

    if args.model_weights_path:
        checkpoint = torch.load(
            args.model_weights_path, map_location=lambda storage, _: storage
        )
        start_epoch = checkpoint.get("epoch", 0)
        best_psnr = checkpoint.get("best_psnr", 0.0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        state_dict = checkpoint.get("state_dict", None)
        if isinstance(model, DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Load model weights from `{args.model_weights_path}` successfully.")
    else:
        print("No model weights path provided, starting from scratch.")

    samples_dir = os.path.join(args.samples_dir, args.model_arch_name)
    results_dir = os.path.join(args.results_dir, args.model_arch_name)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(samples_dir, "logs"))

    scaler_enabled = device.type == "cuda"
    scaler = amp.grad_scaler.GradScaler("cuda", enabled=scaler_enabled)

    psnr_model = PSNR(args.upscale_factor, False).to(device=device)
    ssim_model = SSIM(args.upscale_factor, False).to(device=device)

    for epoch in range(start_epoch, args.epochs):
        train(
            model,
            train_loader,
            criterion,
            optimizer,
            args.upscale_factor,
            epoch,
            scaler,
            writer,
        )
        psnr, ssim = validate(
            model,
            test_loader,
            psnr_model,
            ssim_model,
            epoch,
            args.upscale_factor,
            writer,
            mode="test",
        )
        print("\n")

        scheduler.step()

        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = epoch + 1 == args.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)

        state_dict = (
            model.module.state_dict()
            if isinstance(model, DataParallel)
            else model.state_dict()
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "state_dict": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(samples_dir, f"epoch_{epoch + 1}.pth"),
        )
        if is_best:
            print(f"Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")
            shutil.copyfile(
                os.path.join(samples_dir, f"epoch_{epoch + 1}.pth"),
                os.path.join(results_dir, "best.pth"),
            )
        if is_last:
            shutil.copyfile(
                os.path.join(samples_dir, f"epoch_{epoch + 1}.pth"),
                os.path.join(results_dir, "last.pth"),
            )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a super-resolution model.")
    parser.add_argument("--image_dir", type=str, default="./data/HR")
    parser.add_argument("--samples_dir", type=str, default="./samples")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--upscale_factor", type=float, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--model_milestones", type=list, default=[300, 2400])
    parser.add_argument("--model_gamma", type=float, default=0.1)
    parser.add_argument("--model_arch_name", type=str, default="rt4ksr_x2")
    parser.add_argument("--model_lr", type=float, default=1e-3)
    parser.add_argument("--model_momentum", type=float, default=0.9)
    parser.add_argument("--model_weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_nesterov", type=bool, default=False)
    args = parser.parse_args()
    main(args)
