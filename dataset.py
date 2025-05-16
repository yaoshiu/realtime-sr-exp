import numpy as np
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def image_pipeline(
    wds_paths: list[str],
    crop_size: int,
    upscale_factor: float,
    mode: str = "train",
):
    read = fn.readers.webdataset(
        paths=wds_paths,
        ext=["hr.png", "lr.png"],
        missing_component_behavior="error",
        name="Reader",
    )

    if read is None:
        raise ValueError("No data found in the provided paths.")

    hr, lr = read

    hr = fn.decoders.image_crop(
        hr,
        device="mixed",
    )
    lr = fn.decoders.image_crop(
        lr,
        device="mixed",
    )

    if mode == "train":
        pos_x = fn.random.uniform(range=(0, 1))
        pos_y = fn.random.uniform(range=(0, 1))
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


if __name__ == "__main__":
    test_tar_file_path = "data/test/sr_test-000000.tar"

    # Pipeline parameters
    batch_size = 4
    num_threads = 2
    crop_size_val = 64
    upscale_factor_val = 2.0
    wds_paths_val = [test_tar_file_path]

    print("\n--- Testing 'train' mode ---")
    train_pipe = image_pipeline(
        wds_paths=wds_paths_val,
        crop_size=crop_size_val,
        upscale_factor=upscale_factor_val,
        mode="train",
        batch_size=batch_size,
        num_threads=num_threads,
    )

    train_pipe.build()
    # Check epoch size
    train_epoch_size = train_pipe.epoch_size("Reader")
    print(f"Reader epoch size (train): {train_epoch_size}")
    if train_epoch_size == 0 or not isinstance(train_epoch_size, int):
        print(
            "ERROR: No data found by the WebDataset reader. Check paths and tar file content."
        )
    else:
        num_train_iterations = (train_epoch_size + batch_size - 1) // batch_size
        print(f"Running {num_train_iterations} iterations for train mode...")
        for i in range(num_train_iterations):
            try:
                outputs = train_pipe.run()
                hr_out_gpu = outputs[0]  # This is a TensorListGPU
                lr_out_gpu = outputs[1]  # This is a TensorListGPU

                # Transfer to CPU to check shapes and data (as numpy arrays)
                hr_out_cpu = (
                    hr_out_gpu.as_cpu().as_array()
                )  # .as_array() concatenates tensors in the list
                lr_out_cpu = lr_out_gpu.as_cpu().as_array()

                print(f"\nTrain Iteration {i+1}/{num_train_iterations}:")
                print(
                    f"  HR batch shape: {hr_out_cpu.shape}, dtype: {hr_out_cpu.dtype}"
                )
                print(
                    f"  LR batch shape: {lr_out_cpu.shape}, dtype: {lr_out_cpu.dtype}"
                )
                print(
                    f"  HR min: {np.min(hr_out_cpu):.2f}, max: {np.max(hr_out_cpu):.2f}, mean: {np.mean(hr_out_cpu):.2f}"
                )
                print(
                    f"  LR min: {np.min(lr_out_cpu):.2f}, max: {np.max(lr_out_cpu):.2f}, mean: {np.mean(lr_out_cpu):.2f}"
                )

                # Verify expected shapes
                expected_hr_h = int(crop_size_val * upscale_factor_val)
                expected_hr_w = int(crop_size_val * upscale_factor_val)
                # The actual batch size might be smaller in the last iteration if not perfectly divisible
                current_batch_size = hr_out_cpu.shape[0]
                assert hr_out_cpu.shape == (
                    current_batch_size,
                    3,
                    expected_hr_h,
                    expected_hr_w,
                )
                assert lr_out_cpu.shape == (
                    current_batch_size,
                    3,
                    crop_size_val,
                    crop_size_val,
                )
                assert hr_out_cpu.dtype == np.float32
                assert lr_out_cpu.dtype == np.float32
                assert np.max(hr_out_cpu) <= 1.0 and np.min(hr_out_cpu) >= 0.0
                assert np.max(lr_out_cpu) <= 1.0 and np.min(lr_out_cpu) >= 0.0

            except StopIteration:
                print("StopIteration: End of dataset in train mode.")
                break
            except RuntimeError as e:
                print(f"DALI RuntimeError: {e}")
                if "WebDataset reader" in str(e) and "component not found" in str(e):
                    print(
                        "Hint: Ensure your tar file contains both 'hr.png' and 'lr.png' for each sample key."
                    )
                break
        print("Train mode test finished.")

    print("\n--- Testing 'test' mode ---")
    val_pipe = image_pipeline(
        wds_paths=wds_paths_val,
        crop_size=crop_size_val,
        upscale_factor=upscale_factor_val,
        mode="test",
        batch_size=batch_size,
        num_threads=num_threads,
    )

    val_pipe.build()
    val_epoch_size = val_pipe.epoch_size("Reader")
    print(f"Reader epoch size (val): {val_epoch_size}")

    if val_epoch_size == 0 or not isinstance(val_epoch_size, int):
        print("ERROR: No data found by the WebDataset reader for val mode.")
    else:
        num_val_iterations = (val_epoch_size + batch_size - 1) // batch_size
        print(f"Running {num_val_iterations} iterations for val mode...")
        for i in range(num_val_iterations):
            try:
                outputs = val_pipe.run()
                hr_out_gpu = outputs[0]
                lr_out_gpu = outputs[1]

                hr_out_cpu = hr_out_gpu.as_cpu().as_array()
                lr_out_cpu = lr_out_gpu.as_cpu().as_array()

                print(f"\nTest Iteration {i+1}/{num_val_iterations}:")
                print(
                    f"  HR batch shape: {hr_out_cpu.shape}, dtype: {hr_out_cpu.dtype}"
                )
                print(
                    f"  LR batch shape: {lr_out_cpu.shape}, dtype: {lr_out_cpu.dtype}"
                )
                # For val mode, crops should be centered, so values might be more consistent if images are similar
                print(
                    f"  HR min: {np.min(hr_out_cpu):.2f}, max: {np.max(hr_out_cpu):.2f}, mean: {np.mean(hr_out_cpu):.2f}"
                )
                print(
                    f"  LR min: {np.min(lr_out_cpu):.2f}, max: {np.max(lr_out_cpu):.2f}, mean: {np.mean(lr_out_cpu):.2f}"
                )

            except StopIteration:
                print("StopIteration: End of dataset in val mode.")
                break
            except RuntimeError as e:
                print(f"DALI RuntimeError: {e}")
                break
        print("Test mode test finished.")

    print("\nPipeline testing complete.")
