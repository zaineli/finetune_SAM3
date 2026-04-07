#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python3"
VENV_DIR="${PROJECT_ROOT}/.venv"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
SAM3_REPO_URL="https://github.com/facebookresearch/sam3.git"
SAM3_COMMIT="bfbed072a07a6a52c8d5fdc75a7a186251a835b1"
HF_TOKEN="${HF_TOKEN:-}"
GENERATE_JITTERED_BAR_DATASET=0
JITTERED_BAR_NUM_CHARTS=500
INSTALL_SYSTEM_PACKAGES=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Bootstraps this project on a fresh GPU server, including:
  1. creating a virtual environment
  2. cloning/pinning the upstream SAM3 repo
  3. patching SAM3 so local YAML config paths work
  4. rendering the tracked SAM3 fine-tuning configs into the clone
  5. installing PyTorch, SAM3 training extras, and dataset-generation deps
  6. optionally downloading sam3.pt into the expected checkpoint path

Options:
  --python PATH                Python executable to use (default: python3)
  --venv PATH                  Virtual environment path (default: ${PROJECT_ROOT}/.venv)
  --torch-index-url URL        PyTorch wheel index URL (default: ${TORCH_INDEX_URL})
  --torch-version VERSION      Torch version to install (default: ${TORCH_VERSION})
  --torchvision-version VER    Torchvision version to install (default: ${TORCHVISION_VERSION})
  --hf-token TOKEN             Hugging Face token to immediately download sam3.pt
  --generate-datasets          Force regeneration of the jittered bar dataset only
  --jittered-bar-num-charts N  Number of jittered bar images to generate (default: ${JITTERED_BAR_NUM_CHARTS})
  --install-system-packages    Install Ubuntu/Debian system packages with apt-get when available
  -h, --help                   Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") --hf-token hf_xxx
  HF_TOKEN=hf_xxx $(basename "$0") --generate-datasets
  $(basename "$0") --generate-datasets --jittered-bar-num-charts 500
EOF
}

log() {
  printf '[bootstrap] %s\n' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torchvision-version)
      TORCHVISION_VERSION="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --generate-datasets)
      GENERATE_JITTERED_BAR_DATASET=1
      shift
      ;;
    --jittered-bar-num-charts)
      JITTERED_BAR_NUM_CHARTS="$2"
      shift 2
      ;;
    --install-system-packages)
      INSTALL_SYSTEM_PACKAGES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${INSTALL_SYSTEM_PACKAGES}" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      log "Installing Ubuntu/Debian system packages."
      sudo apt-get update
      sudo apt-get install -y \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        pkg-config \
        python3-dev \
        python3-venv
    else
      log "Skipping system package install because sudo is unavailable."
    fi
  else
    log "Skipping system package install because apt-get is unavailable."
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PROJECT_ROOT

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

log "Upgrading pip tooling."
"${PIP}" install --upgrade pip setuptools wheel

if [[ ! -d "${PROJECT_ROOT}/sam3/.git" ]]; then
  log "Cloning upstream SAM3 into ${PROJECT_ROOT}/sam3"
  git clone "${SAM3_REPO_URL}" "${PROJECT_ROOT}/sam3"
  git -C "${PROJECT_ROOT}/sam3" checkout "${SAM3_COMMIT}"
else
  log "SAM3 clone already exists at ${PROJECT_ROOT}/sam3"
fi

log "Patching SAM3 train entrypoint for local YAML file-path configs."
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
train_py = project_root / "sam3" / "sam3" / "train" / "train.py"
text = train_py.read_text(encoding="utf-8")

if "def compose_train_config(config_arg: str):" not in text:
    original_import = (
        "from argparse import ArgumentParser\n"
        "from copy import deepcopy\n\n"
        "import submitit\n"
        "import torch\n"
        "from hydra import compose, initialize_config_module\n"
    )
    patched_import = (
        "from argparse import ArgumentParser\n"
        "from copy import deepcopy\n"
        "from pathlib import Path\n\n"
        "import submitit\n"
        "import torch\n"
        "from hydra import compose, initialize_config_dir, initialize_config_module\n"
    )
    if original_import not in text:
        raise RuntimeError("Unable to patch SAM3 imports in train.py; upstream file layout changed.")
    text = text.replace(original_import, patched_import, 1)

    original_main = (
        "\n\ndef main(args) -> None:\n"
        "    cfg = compose(config_name=args.config)\n"
    )
    patched_main = (
        "\n\ndef compose_train_config(config_arg: str):\n"
        "    config_path = Path(config_arg)\n"
        "    if config_path.suffix in {\".yaml\", \".yml\"} and config_path.exists():\n"
        "        with initialize_config_dir(\n"
        "            config_dir=str(config_path.parent.resolve()), version_base=\"1.2\"\n"
        "        ):\n"
        "            return compose(config_name=config_path.name)\n\n"
        "    with initialize_config_module(\"sam3.train\", version_base=\"1.2\"):\n"
        "        return compose(config_name=config_arg)\n\n\n"
        "def main(args) -> None:\n"
        "    cfg = compose_train_config(args.config)\n"
    )
    if original_main not in text:
        raise RuntimeError("Unable to patch SAM3 main() config loading; upstream file layout changed.")
    text = text.replace(original_main, patched_main, 1)

    old_bootstrap = (
        "if __name__ == \"__main__\":\n"
        "    initialize_config_module(\"sam3.train\", version_base=\"1.2\")\n"
        "    parser = ArgumentParser()\n"
    )
    new_bootstrap = (
        "if __name__ == \"__main__\":\n"
        "    parser = ArgumentParser()\n"
    )
    if old_bootstrap not in text:
        raise RuntimeError("Unable to patch SAM3 __main__ bootstrap; upstream file layout changed.")
    text = text.replace(old_bootstrap, new_bootstrap, 1)

    train_py.write_text(text, encoding="utf-8")
    print(f"Patched {train_py}")
else:
    print(f"Patch already present in {train_py}")
PY

log "Rendering tracked SAM3 config templates into the clone."
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
template_dir = project_root / "sam3_config_templates" / "roboflow_v100"
target_dir = project_root / "sam3" / "train" / "configs" / "roboflow_v100"
target_dir.mkdir(parents=True, exist_ok=True)

for template_path in sorted(template_dir.glob("*.yaml.template")):
    rendered_text = template_path.read_text(encoding="utf-8").replace(
        "__PROJECT_ROOT__", str(project_root)
    )
    target_path = target_dir / template_path.name.replace(".template", "")
    target_path.write_text(rendered_text, encoding="utf-8")
    print(f"Rendered {target_path}")
PY

log "Installing PyTorch wheels from ${TORCH_INDEX_URL}"
"${PIP}" install --upgrade --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

log "Installing SAM3 training extras."
(cd "${PROJECT_ROOT}/sam3" && "${PIP}" install -e '.[train]')

log "Installing dataset-generation and COCO tooling."
"${PIP}" install \
  "numpy>=1.26,<2" \
  matplotlib \
  opencv-python-headless \
  pillow \
  pycocotools \
  scipy

log "Running an import smoke test."
"${PYTHON}" - <<'PY'
import cv2
import matplotlib
import pycocotools.mask
import sam3
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
print("matplotlib:", matplotlib.__version__)
print("sam3:", sam3.__version__)
PY

generate_jittered_bar_dataset() {
  log "Generating jittered bar dataset with ${JITTERED_BAR_NUM_CHARTS} images."
  (cd "${PROJECT_ROOT}" && "${PYTHON}" jittered_bar_script.py --num-charts "${JITTERED_BAR_NUM_CHARTS}" --skip-preview)
}

if [[ "${GENERATE_JITTERED_BAR_DATASET}" -eq 1 ]]; then
  generate_jittered_bar_dataset
else
  if [[ ! -f "${PROJECT_ROOT}/chart_dataset/annotations/train.json" ]]; then
    log "chart_dataset is missing; generating the required jittered bar dataset automatically."
    generate_jittered_bar_dataset
  else
    log "chart_dataset already exists; skipping dataset generation."
  fi
fi

if [[ -n "${HF_TOKEN}" ]]; then
  log "Downloading gated SAM3 checkpoint into the pipeline checkpoint path."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/download_sam3_checkpoint.py" \
    --token "${HF_TOKEN}" \
    --also-download-config \
    --validate \
    --validate-config chart_dataset_ft.yaml
else
  log "Skipping checkpoint download because no HF token was provided."
fi

log "Bootstrap complete."
log "Activate the environment with: source ${VENV_DIR}/bin/activate"
log "Next step: ${PYTHON} ${PROJECT_ROOT}/scripts/download_sam3_checkpoint.py --token YOUR_HF_TOKEN --validate --validate-config chart_dataset_ft.yaml"
log "Training entrypoint: cd ${PROJECT_ROOT}/sam3 && python sam3/train/train.py -c train/configs/roboflow_v100/chart_dataset_ft.yaml --use-cluster 0 --num-gpus 1"
