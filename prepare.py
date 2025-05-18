import concurrent.futures
from glob import glob
import os
import concurrent
import cv2
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.data_node import DataNode
from nvidia.dali import fn, types
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import webdataset
import argparse


@pipeline_def
def image_scale_pipeline(files: list[str], scale: float, seed: int = 42):
    file, _ = fn.readers.file(name="Reader", files=files, seed=seed)

    hr = fn.decoders.image(
        file,
        device="mixed",
        output_type=types.DALIImageType.BGR,
    )

    hr_size = hr.shape()[0:2]

    lr_size = hr_size / scale

    lr: DataNode = fn.resize(hr, size=lr_size, interp_type=types.DALIInterpType.INTERP_CUBIC, device="gpu")  # type: ignore

    return hr, lr


def main(args):
    print("Preparing dataset...")

    files = [
        file
        for directory in args.dirs
        for ext in args.ext
        for file in glob(os.path.join(directory, f"*.{ext}"))
    ]

    if not files:
        raise ValueError("No files found in the provided directories.")

    train, test = train_test_split(
        files,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(f"Found {len(train)} training files and {len(test)} testing files.")

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_pattern = os.path.join(train_dir, "sr_train-%06d.tar")
    test_pattern = os.path.join(test_dir, "sr_test-%06d.tar")

    print("Creating training dataset...")
    make_wds(
        files=train,
        pattern=train_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scale=args.scale,
        maxcount=args.maxcount,
        seed=args.seed,
    )

    print("Creating testing dataset...")
    make_wds(
        files=test,
        pattern=test_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scale=args.scale,
        maxcount=args.maxcount,
        seed=args.seed + 1,
    )


def make_wds(
    files: list[str],
    pattern: str,
    batch_size: int,
    num_workers: int,
    scale: float,
    maxcount: int = 1000,
    seed: int = 42,
):
    pipe = image_scale_pipeline(
        files=files,
        batch_size=batch_size,
        num_threads=num_workers,
        exec_dynamic=True,
        scale=scale,
        seed=seed,
    )
    pipe.build()

    with webdataset.ShardWriter(
        pattern, maxcount
    ) as sink, concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
        for epoch in tqdm(
            range((len(files) + batch_size - 1) // batch_size), desc="Epochs"
        ):
            hrs, lrs = pipe.run()

            hrs = hrs.as_cpu()
            lrs = lrs.as_cpu()

            futures = []
            for i in range(len(hrs)):
                hr = hrs.at(i)
                lr = lrs.at(i)

                index = epoch * batch_size + i

                if index >= len(files):
                    break

                key = f"{index:06d}"

                future = executor.submit(_encode_image, key, hr, lr)

                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    sink.write(result)
                except Exception as e:
                    print(f"Error processing image: {e}")


def _encode_image(key, hr, lr):
    hr_success, hr_buffer = cv2.imencode(".png", hr)
    lr_success, lr_buffer = cv2.imencode(".png", lr)

    if not hr_success or not lr_success:
        raise ValueError("Failed to encode image.")

    return {
        "__key__": key,
        "hr.png": hr_buffer.tobytes(),
        "lr.png": lr_buffer.tobytes(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for DALI.")
    parser.add_argument(
        "dirs",
        type=str,
        nargs="+",
        help="Directories containing the images.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the prepared dataset.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=["png", "jpg", "jpeg"],
        help="Image file extensions to include.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Scale factor for downsampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the pipeline.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for the pipeline.",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Maximum number of samples per shard.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and splitting the dataset.",
    )

    args = parser.parse_args()

    main(args)
