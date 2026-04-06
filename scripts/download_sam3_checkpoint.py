#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


DEFAULT_REPO_ID = "facebook/sam3"
DEFAULT_FILENAME = "sam3.pt"
DATASET_CONFIGS = (
    "chart_dataset_ft.yaml",
    "volcano_dataset_ft.yaml",
    "boxplot_dataset_ft.yaml",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_output = project_root / "sam3" / "checkpoints" / DEFAULT_FILENAME

    parser = argparse.ArgumentParser(
        description=(
            "Download the gated SAM3 checkpoint from Hugging Face into the exact "
            "path expected by this fine-tuning pipeline."
        )
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token with approved access to facebook/sam3. Falls back to HF_TOKEN.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository to pull from. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Checkpoint filename to download. Default: {DEFAULT_FILENAME}",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Destination path for the checkpoint file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download even if the checkpoint already exists at the destination.",
    )
    parser.add_argument(
        "--also-download-config",
        action="store_true",
        help="Also download config.json next to the checkpoint for completeness.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "After download, validate the local fine-tuning setup by resolving configs, "
            "loading datasets, loading the checkpoint, and running one smoke forward/loss pass."
        ),
    )
    return parser.parse_args()


def validate_local_setup(project_root: Path) -> None:
    import torch
    from hydra.utils import instantiate

    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.train import compose_train_config

    config_dir = project_root / "sam3" / "train" / "configs" / "roboflow_v100"
    for config_name in DATASET_CONFIGS:
        config_path = config_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        cfg = compose_train_config(str(config_path))
        dataset = instantiate(cfg.trainer.data.train.dataset, _convert_="all")
        collate = instantiate(cfg.trainer.data.train.collate_fn, _convert_="all")
        sample = dataset[0]
        batch = collate([sample])
        if "all" not in batch:
            raise RuntimeError(f"Unexpected collate output for {config_name}: {batch.keys()}")
        print(f"Validated dataset loading for {config_name}: len={len(dataset)}")

    chart_cfg = compose_train_config(str(config_dir / "chart_dataset_ft.yaml"))
    model = instantiate(chart_cfg.trainer.model, _convert_="all")
    loss_fn = instantiate(chart_cfg.trainer.loss.all, _convert_="all")
    dataset = instantiate(chart_cfg.trainer.data.train.dataset, _convert_="all")
    collate = instantiate(chart_cfg.trainer.data.train.collate_fn, _convert_="all")
    batch = collate([dataset[0]])["all"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    batch = copy_data_to_device(batch, device, non_blocking=False)

    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_context:
        find_stages = model(batch)
        find_targets = [model.back_convert(x) for x in batch.find_targets]
        loss_dict = loss_fn(find_stages, find_targets)

    core_loss = loss_dict["core_loss"]
    if not torch.isfinite(core_loss):
        raise FloatingPointError(f"Non-finite core loss during smoke validation: {core_loss.item()}")

    if device.type == "cuda":
        core_loss.backward()
        model.zero_grad(set_to_none=True)

    print(
        "Smoke validation passed: "
        f"device={device.type}, core_loss={core_loss.item():.6f}, "
        f"loss_terms={sorted(loss_dict.keys())[:6]}..."
    )


def download_hf_file(
    *,
    repo_id: str,
    filename: str,
    token: str,
    output_path: Path,
    force: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"Checkpoint already present at: {output_path}")
        return output_path

    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=str(output_path.parent),
            force_download=force,
        )
    )

    if downloaded_path.resolve() != output_path.resolve():
        shutil.copy2(downloaded_path, output_path)

    return output_path


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Missing Hugging Face token. Pass --token or set HF_TOKEN in the environment.",
            file=sys.stderr,
        )
        return 2

    output_path = Path(args.output).expanduser().resolve()

    try:
        checkpoint_path = download_hf_file(
            repo_id=args.repo_id,
            filename=args.filename,
            token=token,
            output_path=output_path,
            force=args.force,
        )
        print(f"Downloaded checkpoint to: {checkpoint_path}")

        if args.also_download_config:
            config_path = output_path.parent / "config.json"
            download_hf_file(
                repo_id=args.repo_id,
                filename="config.json",
                token=token,
                output_path=config_path,
                force=args.force,
            )
            print(f"Downloaded config to: {config_path}")

        if args.validate:
            project_root = Path(__file__).resolve().parents[1]
            validate_local_setup(project_root)
    except GatedRepoError as error:
        print(
            "Access to the Hugging Face repo is still gated. Make sure the token belongs to "
            "an account that has been approved for facebook/sam3.",
            file=sys.stderr,
        )
        print(str(error), file=sys.stderr)
        return 3
    except RepositoryNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 4
    except HfHubHTTPError as error:
        print(f"Hugging Face download failed: {error}", file=sys.stderr)
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
