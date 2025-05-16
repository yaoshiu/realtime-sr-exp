import os
import argparse
import random
from glob import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import webdataset as wds
from tqdm import tqdm
import multiprocessing
from functools import partial
import io

# PyTorch for GPU acceleration (optional)
try:
    import torch
    import torchvision.transforms.functional as TF

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print(
        "PyTorch not found. GPU acceleration for resizing will be disabled. Using OpenCV for resizing."
    )


def get_image_paths(input_dir):
    """Finds all supported image paths in a directory."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_dir, ext)))
        image_paths.extend(
            glob(os.path.join(input_dir, "**", ext), recursive=True)
        )  # Include subdirectories
    return sorted(list(set(image_paths)))  # Sort and remove duplicates


def resize_image_torch(img_np_rgb, scale_factor, device):
    """Resizes an image using PyTorch on GPU or CPU."""
    # HWC (NumPy) -> CHW (Tensor)
    img_tensor = torch.from_numpy(img_np_rgb.transpose((2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(
        device
    )  # Add batch dimension and send to device

    h, w = img_np_rgb.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    if lr_h == 0 or lr_w == 0:  # Avoid zero dimensions
        print(
            f"Warning: LR dimensions would be zero for scale {scale_factor} on image of size {w}x{h}. Skipping resize."
        )
        return None

    # Bicubic interpolation with antialiasing is generally good for downscaling
    lr_tensor = TF.resize(
        img_tensor,
        [lr_h, lr_w],
        interpolation=TF.InterpolationMode.BICUBIC,
        antialias=True,
    )

    # CHW (Tensor) -> HWC (NumPy)
    lr_img_np_rgb = (
        (lr_tensor * 255.0)
        .clamp(0, 255)
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose((1, 2, 0))
        .astype(np.uint8)
    )
    return lr_img_np_rgb


def resize_image_cv2(img_np_rgb, scale_factor):
    """Resizes an image using OpenCV on CPU."""
    h, w = img_np_rgb.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    if lr_h == 0 or lr_w == 0:
        print(
            f"Warning: LR dimensions would be zero for scale {scale_factor} on image of size {w}x{h}. Skipping resize."
        )
        return None
    # INTER_AREA is good for shrinking, INTER_CUBIC for quality
    lr_img_np_rgb = cv2.resize(img_np_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    return lr_img_np_rgb


def process_image(hr_path, scale_factor, use_gpu, device_str):
    """
    Loads an HR image, generates LR, and returns them as bytes.
    This function is designed to be used with multiprocessing.Pool.
    """
    try:
        # 1. Load HR image
        hr_img_bgr = cv2.imread(hr_path)
        if hr_img_bgr is None:
            print(f"Warning: Could not read image {hr_path}. Skipping.")
            return None
        hr_img_rgb = cv2.cvtColor(hr_img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Generate LR image
        lr_img_rgb = None
        if use_gpu and TORCH_AVAILABLE:
            # Note: each process needs its own device context if passing device strings
            # For simplicity in multiprocessing, we often pass device string and create tensor on device inside process
            # Or, we can initialize the device per worker (more complex)
            # For this script, passing device string is fine as TF.resize handles it.
            # PyTorch GIL release for C++ ops means this can parallelize well.
            device = torch.device(device_str)  # Create device object inside worker
            lr_img_rgb = resize_image_torch(hr_img_rgb, scale_factor, device)
        else:
            lr_img_rgb = resize_image_cv2(hr_img_rgb, scale_factor)

        if lr_img_rgb is None:  # Resize failed (e.g., zero dimensions)
            return None

        # 3. Encode images to PNG bytes
        # For cv2.imencode, it expects BGR, so convert back if needed.
        _, hr_bytes = cv2.imencode(".png", cv2.cvtColor(hr_img_rgb, cv2.COLOR_RGB2BGR))
        _, lr_bytes = cv2.imencode(".png", cv2.cvtColor(lr_img_rgb, cv2.COLOR_RGB2BGR))

        # 4. Create a unique key (e.g., filename without extension)
        key = os.path.splitext(os.path.basename(hr_path))[0]

        return {
            "__key__": key,
            "hr.png": hr_bytes.tobytes(),
            "lr.png": lr_bytes.tobytes(),
        }
    except Exception as e:
        print(f"Error processing {hr_path}: {e}")
        return None


def create_webdataset(
    image_paths,
    output_pattern,
    scale_factor,
    use_gpu,
    device_str,
    num_workers,
    shard_size,
    set_name,
):
    """
    Creates a WebDataset from a list of image paths.
    """
    print(f"\nProcessing {set_name} set: {len(image_paths)} images...")
    os.makedirs(os.path.dirname(output_pattern), exist_ok=True)

    # Use a partial function to pass fixed arguments to process_image
    # The device_str is passed, and torch.device is created inside each worker process.
    # This is safer for multiprocessing with CUDA.
    task_fn = partial(
        process_image, scale_factor=scale_factor, use_gpu=use_gpu, device_str=device_str
    )

    with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
        if num_workers > 0 and len(image_paths) > 1:  # Multiprocessing
            # Using imap_unordered for potentially faster yield if processing times vary
            with multiprocessing.Pool(processes=num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(task_fn, image_paths),
                    total=len(image_paths),
                    desc=f"Creating {set_name} WebDataset",
                ):
                    if result:
                        sink.write(result)
        else:  # Single process (for debugging or small datasets)
            print("Using single process mode.")
            for hr_path in tqdm(image_paths, desc=f"Creating {set_name} WebDataset"):
                result = task_fn(hr_path)
                if result:
                    sink.write(result)
    print(
        f"{set_name} WebDataset creation complete. Shards saved to pattern: {output_pattern}"
    )


def main(args):
    if args.num_workers is None:
        args.num_workers = (
            os.cpu_count() or 1
        )  # Default to all CPUs, or 1 if cpu_count() is None

    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.gpu and not TORCH_AVAILABLE:
        print(
            "Warning: --gpu flag was set, but PyTorch is not available. Falling back to CPU resizing."
        )
        args.gpu = False

    device_str = args.gpu_device if args.gpu else "cpu"
    if args.gpu and TORCH_AVAILABLE:
        if not torch.cuda.is_available():
            print(
                f"Warning: --gpu flag was set, but CUDA is not available on device {device_str}. Falling back to CPU resizing."
            )
            args.gpu = False
            device_str = "cpu"
        else:
            try:
                # Test device accessibility
                _ = torch.tensor([1.0]).to(device_str)
                print(f"Using PyTorch on device: {device_str} for resizing.")
            except Exception as e:
                print(
                    f"Error accessing device {device_str}: {e}. Falling back to CPU resizing."
                )
                args.gpu = False
                device_str = "cpu"

    if not args.gpu:
        print("Using OpenCV on CPU for resizing.")

    # 1. Get all image paths
    all_hr_paths = get_image_paths(args.input_dir)
    if not all_hr_paths:
        print(f"No images found in {args.input_dir}. Exiting.")
        return
    print(f"Found {len(all_hr_paths)} total images.")

    # 2. Split into training and testing sets
    if args.test_size > 0 and args.test_size < 1.0:
        train_paths, test_paths = train_test_split(
            all_hr_paths, test_size=args.test_size, random_state=args.random_seed
        )
        print(
            f"Split: {len(train_paths)} training images, {len(test_paths)} testing images."
        )
    elif args.test_size == 0:
        train_paths = all_hr_paths
        test_paths = []
        print(
            f"Using all {len(train_paths)} images for training, no test set created by this script."
        )
    elif args.test_size >= 1.0:  # Treat as number of test samples if >= 1
        if int(args.test_size) >= len(all_hr_paths):
            print(
                "Warning: test_size is greater than or equal to total images. All images will be in test set."
            )
            train_paths = []
            test_paths = all_hr_paths
        else:
            train_paths, test_paths = train_test_split(
                all_hr_paths,
                test_size=int(args.test_size),  # interpret as count
                random_state=args.random_seed,
            )
        print(
            f"Split: {len(train_paths)} training images, {len(test_paths)} testing images."
        )
    else:  # test_size < 0 is invalid
        print(
            "Invalid test_size. Must be between 0.0 and 1.0, or an integer count >= 1."
        )
        return

    # 3. Create WebDatasets
    # Train set
    if train_paths:
        train_output_pattern = os.path.join(
            args.output_dir, "train", "sr_train-%06d.tar"
        )
        create_webdataset(
            train_paths,
            train_output_pattern,
            args.scale_factor,
            args.gpu,
            device_str,
            args.num_workers,
            args.shard_size,
            "Training",
        )

    # Test set
    if test_paths:
        test_output_pattern = os.path.join(args.output_dir, "test", "sr_test-%06d.tar")
        create_webdataset(
            test_paths,
            test_output_pattern,
            args.scale_factor,
            args.gpu,
            device_str,
            args.num_workers,
            args.shard_size,
            "Testing",
        )

    print("\nAll processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LR images and create WebDataset for Super-Resolution."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing high-resolution images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save WebDataset shards.",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=2,
        help="Downscaling factor for LR images (e.g., 4 for x4 SR).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (0.0 to 1.0).",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for train/test split."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to os.cpu_count(). Set to 0 for single process.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Maximum number of samples per shard file.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for resizing if PyTorch is available.",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="cuda:0",
        help="GPU device to use (e.g., 'cuda:0', 'cuda:1'). Ignored if --gpu is not set or PyTorch not available.",
    )

    args = parser.parse_args()

    # Important for multiprocessing on Windows, and good practice elsewhere
    multiprocessing.freeze_support()

    if args.gpu and TORCH_AVAILABLE:
        try:
            if (
                multiprocessing.get_start_method(allow_none=True) is None
                or multiprocessing.get_start_method() != "spawn"
            ):
                multiprocessing.set_start_method("spawn", force=True)
                print(
                    "Set multiprocessing start method to 'spawn' for CUDA compatibility."
                )
            elif multiprocessing.get_start_method() == "spawn":
                print("Multiprocessing start method already 'spawn'.")

        except RuntimeError as e:
            print(
                f"Warning: Could not set multiprocessing start method to 'spawn': {e}"
            )
            print("If using GPU with PyTorch, this might lead to errors.")

    main(args)
