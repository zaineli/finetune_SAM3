import json
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


def default_preview_root(output_root: Path) -> Path:
    return output_root.with_name(f"{output_root.name}_preview")


def prepare_preview_dirs(preview_root: Path, clean_output: bool):
    if clean_output and preview_root.exists():
        shutil.rmtree(preview_root)

    for relative_dir in (
        "images/train",
        "images/val",
        "masks/train",
        "masks/val",
        "masked/train",
        "masked/val",
        "annotations",
    ):
        (preview_root / relative_dir).mkdir(parents=True, exist_ok=True)


def save_preview_assets(
    preview_root: Path,
    split_name: str,
    image_filename: str,
    rgb_image: np.ndarray,
    class_masks: dict[str, np.ndarray],
):
    mpimg.imsave(preview_root / "images" / split_name / image_filename, rgb_image)

    stem = Path(image_filename).stem
    for class_label, binary_mask in class_masks.items():
        if binary_mask is None:
            continue
        binary_mask = binary_mask.astype(np.uint8)
        if int(binary_mask.sum()) == 0:
            continue

        mask_filename = f"{stem}_{class_label}.png"
        mpimg.imsave(
            preview_root / "masks" / split_name / mask_filename,
            binary_mask * 255,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        masked_rgb = np.zeros_like(rgb_image)
        mask_bool = binary_mask.astype(bool)
        masked_rgb[mask_bool] = rgb_image[mask_bool]
        mpimg.imsave(preview_root / "masked" / split_name / mask_filename, masked_rgb)


def write_preview_annotations(preview_root: Path, train_json: dict, val_json: dict):
    with (preview_root / "annotations" / "train.json").open("w", encoding="utf-8") as handle:
        json.dump(train_json, handle, indent=2)
    with (preview_root / "annotations" / "val.json").open("w", encoding="utf-8") as handle:
        json.dump(val_json, handle, indent=2)
