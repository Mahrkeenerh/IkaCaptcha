#!/usr/bin/env bash
#
# Download and extract iKaptcha training datasets from the v1.0.0 GitHub Release.
# Run from the repo root:  bash scripts/download_data.sh
#
# Windows users without a POSIX shell: grab the .zip archives manually from
# https://github.com/Mahrkeenerh/iKaptcha/releases/tag/v1.0.0 and extract them
# into data/ (the archives are named with their target directory as the root).

set -euo pipefail

REPO="Mahrkeenerh/iKaptcha"
TAG="v1.0.0"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

fetch_and_extract () {
    local name="$1"
    local archive="${name}.tar.gz"
    echo ">> ${archive}"
    if [[ -d "${name}" ]]; then
        echo "   (already extracted, skipping)"
        return
    fi
    curl -fL --progress-bar -o "${archive}" "${BASE}/${archive}"
    tar -xzf "${archive}"
    rm "${archive}"
}

fetch_and_extract ikariam_pirate_captcha_dataset
fetch_and_extract dataset_pseudo_v2

echo
echo "Done. Datasets ready in $(cd "${DATA_DIR}" && pwd):"
echo "  - ikariam_pirate_captcha_dataset/   (original 1,200/300 YOLO dataset)"
echo "  - dataset_pseudo_v2/                (production 11,210/298 train/val)"
