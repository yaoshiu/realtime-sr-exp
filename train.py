from glob import glob
import os
import random
import shutil
import time
import argparse
import torch
from torch import amp, optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import DataParallel
from dataset import image_pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from utils import *


def load_dataset(
    train_dir: str,
    test_dir: str,
    *,
    upscale_factor: float,
    batch_size=64,
    crop_size=256,
    shuffle=True,
    num_workers=32,
    seed=42,
):
    tr_paths = sorted(glob(os.path.join(train_dir, "*.tar")))
    te_paths = sorted(glob(os.path.join(test_dir, "*.tar")))

    tr_pipeline = image_pipeline(
        wds_paths=tr_paths,
        crop_size=crop_size,
        upscale_factor=upscale_factor,
        mode="train",
        shuffle=shuffle,
        batch_size=batch_size,
        num_threads=num_workers,
        seed=seed,
    )
    te_pipeline = image_pipeline(
        wds_paths=te_paths,
        crop_size=crop_size,
        upscale_factor=upscale_factor,
        mode="test",
        shuffle=False,
        batch_size=batch_size,
        num_threads=num_workers,
        seed=seed,
    )
    tr_pipeline.build()
    te_pipeline.build()

    tr_loader = DALIGenericIterator(
        tr_pipeline,
        ["hr", "lr"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )
    te_loader = DALIGenericIterator(
        te_pipeline,
        ["hr", "lr"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    return tr_loader, te_loader


def train(
    model: torch.nn.Module,
    train_loader: DALIGenericIterator,
    criterion: torch.nn.MSELoss,
    optimizer: optim.Optimizer,
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

    end = time.time()

    for i, d in enumerate(train_loader):
        data_time.update(time.time() - end)

        model.zero_grad()

        with amp.autocast_mode.autocast("cuda"):
            hr = d[0]["hr"]
            lr = d[0]["lr"]

            hr = rgb_to_y_torch(hr)
            lr = rgb_to_y_torch(lr)

            sr = model(lr)

            loss = criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), lr.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            writer.add_scalar("Train/Loss", loss.item(), i + epoch * batches + 1)
            progress.display(i + 1)


def validate(
    model: torch.nn.Module,
    valid_loader: DALIGenericIterator,
    psnr_model: PSNR,
    ssim_model: SSIM,
    epoch: int,
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

    end = time.time()

    for i, d in enumerate(valid_loader):
        with torch.no_grad():
            with amp.autocast_mode.autocast("cuda"):
                hr = d[0]["hr"]
                lr = d[0]["lr"]

                lr = rgb_to_y_torch(lr)
                hr = rgb_to_y_torch(hr)

                sr = model(lr)

            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)

            psnres.update(psnr.mean().item(), lr.size(0))
            ssimes.update(ssim.mean().item(), lr.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
    writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)

    return psnres.avg, ssimes.avg


def main(args):
    start_epoch = 0

    best_psnr = 0.0
    best_ssim = 0.0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = choice_device(args.device)
    print(f"Using `{device}` device.")

    model = build_model(args.model_arch_name)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device=device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    train_loader, test_loader = load_dataset(
        args.train_dir,
        args.test_dir,
        upscale_factor=args.upscale_factor,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print("Load datasets successfully.")

    criterion = torch.nn.MSELoss().to(device=device)
    print("Define loss function successfully.")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.model_lr,
        weight_decay=args.model_weight_decay,
    )
    print("Define optimizer successfully.")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    print("Define scheduler successfully.")

    scaler = amp.grad_scaler.GradScaler()

    if args.model_weights_path:
        checkpoint = torch.load(
            args.model_weights_path, map_location=lambda storage, _: storage
        )
        start_epoch = checkpoint.get("epoch", 0)
        best_psnr = checkpoint.get("best_psnr", 0.0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        state_dict = checkpoint.get("state_dict", None)
        if isinstance(model, DataParallel):
            model.module.load_state_dict(state_dict, strict=True)
        else:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"Load model weights from `{args.model_weights_path}` successfully.")
    else:
        print("No model weights path provided, starting from scratch.")

    samples_dir = os.path.join(
        args.samples_dir, args.model_arch_name, args.experiment_name
    )
    results_dir = os.path.join(
        args.results_dir, args.model_arch_name, args.experiment_name
    )
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(samples_dir, "logs"))

    psnr_model = PSNR(args.upscale_factor, False)
    ssim_model = SSIM(args.upscale_factor, False)

    for epoch in range(start_epoch, args.epochs):
        train(
            model,
            train_loader,
            criterion,
            optimizer,
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
            writer,
            mode="test",
        )
        print("\n")

        scheduler.step()

        is_best = psnr >= best_psnr and ssim >= best_ssim
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
                "scaler": scaler.state_dict(),
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
    parser.add_argument(
        "--experiment_name", type=str, default="AdamW_CosineAnnealingLR"
    )
    parser.add_argument("--train_dir", type=str, default="/dev/shm/train")
    parser.add_argument("--test_dir", type=str, default="/dev/shm/test")
    parser.add_argument("--samples_dir", type=str, default="./samples")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--upscale_factor", type=float, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--model_arch_name", type=str, default="espcn_x2")
    parser.add_argument("--model_lr", type=float, default=2e-4)
    parser.add_argument("--model_weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--crop_size", type=int, default=256, help="Desired LR crop size"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    main(args)
