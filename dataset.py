from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import cv2
import numpy as np


@pipeline_def
def image_pipeline(
    wds_paths: list[str],
    crop_size: int,
    upscale_factor: float,
    mode: str = "train",
    shuffle: bool = True,
    seed: int = 42,
):
    read = fn.readers.webdataset(
        paths=wds_paths,
        ext=["hr.png", "lr.png"],
        missing_component_behavior="error",
        random_shuffle=shuffle,
        seed=seed,
        name="Reader",
    )

    if read is None:
        raise ValueError("No data found in the provided paths.")

    hr, lr = read

    hr = fn.decoders.image(
        hr,
        device="mixed",
    )
    lr = fn.decoders.image(
        lr,
        device="mixed",
    )

    if mode == "train":
        pos_x = fn.random.uniform(range=(0, 1), seed=seed)
        pos_y = fn.random.uniform(range=(0, 1), seed=seed + 1)
    else:
        pos_x = 0.5
        pos_y = 0.5

    hr = fn.crop_mirror_normalize(
        hr,
        dtype=types.DALIDataType.FLOAT,
        scale=1.0 / 255.0,
        crop=(crop_size * upscale_factor, crop_size * upscale_factor),
        crop_pos_x=pos_x,
        crop_pos_y=pos_y,
        device="gpu",
    )
    lr = fn.crop_mirror_normalize(
        lr,
        dtype=types.DALIDataType.FLOAT,
        scale=1.0 / 255.0,
        crop=(crop_size, crop_size),
        crop_pos_x=pos_x,
        crop_pos_y=pos_y,
        device="gpu",
    )

    return hr, lr


# --- Main Test Script ---
if __name__ == "__main__":
    wds_files = ["./data/test/sr_test-000000.tar"]

    # 2. Define pipeline parameters
    batch_size = 2  # Test with a batch size > 1
    crop_s = 256  # Desired LR crop size
    up_factor = 2.0  # Upscale factor

    # --- Test "train" mode (random crops) ---
    print("\n--- Testing 'train' mode (random crops) ---")
    pipe_train = image_pipeline(
        wds_paths=wds_files,
        crop_size=crop_s,
        upscale_factor=up_factor,
        mode="train",
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
    )
    pipe_train.build()

    # Run a few iterations to see different random crops
    for iteration in range(2):  # Show 2 batches
        print(f"Train mode - Iteration {iteration + 1}")
        pipe_out_train = pipe_train.run()

        hr_images_gpu = pipe_out_train[0]
        lr_images_gpu = pipe_out_train[1]

        for i in range(batch_size):
            hr_img_chw = hr_images_gpu.as_cpu().at(i)  # Output is CHW
            lr_img_chw = lr_images_gpu.as_cpu().at(i)  # Output is CHW

            # Convert CHW (DALI output) to HWC (OpenCV input) and scale back to 0-255
            hr_img_hwc = (np.transpose(hr_img_chw, (1, 2, 0)) * 255).astype(np.uint8)
            lr_img_hwc = (np.transpose(lr_img_chw, (1, 2, 0)) * 255).astype(np.uint8)

            # DALI decodes to RGB, OpenCV imshow expects BGR
            hr_img_bgr = cv2.cvtColor(hr_img_hwc, cv2.COLOR_RGB2BGR)
            lr_img_bgr = cv2.cvtColor(lr_img_hwc, cv2.COLOR_RGB2BGR)

            cv2.imshow(f"Train HR Batch {i} Iter {iteration}", hr_img_bgr)
            cv2.imshow(f"Train LR Batch {i} Iter {iteration}", lr_img_bgr)

            print(
                f"  Displayed Train HR shape: {hr_img_bgr.shape}, LR shape: {lr_img_bgr.shape}"
            )

            # Check sizes
            expected_hr_h = int(crop_s * up_factor)
            expected_hr_w = int(crop_s * up_factor)
            expected_lr_h = crop_s
            expected_lr_w = crop_s
            assert hr_img_bgr.shape == (
                expected_hr_h,
                expected_hr_w,
                3,
            ), f"HR shape mismatch: {hr_img_bgr.shape}"
            assert lr_img_bgr.shape == (
                expected_lr_h,
                expected_lr_w,
                3,
            ), f"LR shape mismatch: {lr_img_bgr.shape}"

        print("Press any key to continue to next batch/mode...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Test "validation" mode (center crops) ---
    print("\n--- Testing 'val' mode (center crops) ---")
    pipe_val = image_pipeline(
        wds_paths=wds_files,
        crop_size=crop_s,
        upscale_factor=up_factor,
        mode="val",  # Or any string other than "train"
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
    )
    pipe_val.build()
    pipe_out_val = pipe_val.run()  # Only need one run for center crop verification

    hr_images_gpu_val = pipe_out_val[0]
    lr_images_gpu_val = pipe_out_val[1]

    for i in range(batch_size):
        hr_img_chw_val = hr_images_gpu_val.as_cpu().at(i)
        lr_img_chw_val = lr_images_gpu_val.as_cpu().at(i)

        hr_img_hwc_val = (np.transpose(hr_img_chw_val, (1, 2, 0)) * 255).astype(
            np.uint8
        )
        lr_img_hwc_val = (np.transpose(lr_img_chw_val, (1, 2, 0)) * 255).astype(
            np.uint8
        )

        hr_img_bgr_val = cv2.cvtColor(hr_img_hwc_val, cv2.COLOR_RGB2BGR)
        lr_img_bgr_val = cv2.cvtColor(lr_img_hwc_val, cv2.COLOR_RGB2BGR)

        cv2.imshow(f"Val HR Batch {i}", hr_img_bgr_val)
        cv2.imshow(f"Val LR Batch {i}", lr_img_bgr_val)

        print(
            f"  Displayed Val HR shape: {hr_img_bgr_val.shape}, LR shape: {lr_img_bgr_val.shape}"
        )

    print("Press any key to close all windows and exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nTest finished.")
