#!/bin/bash
# Download COCO 2017 val set (annotations + images) for FID/CLIP evaluation.
# Uses Python's zipfile instead of unzip (not available on all compute nodes).
set -euo pipefail

COCO_DIR="${1:-$(dirname $(dirname $(realpath $0)))/data/coco2017}"
mkdir -p "$COCO_DIR"

PY="${PYTHON_BIN:-python3}"

# Annotations (captions)
if [[ ! -f "$COCO_DIR/annotations/captions_val2017.json" ]]; then
    echo "Downloading COCO 2017 annotations..."
    wget -q -O "$COCO_DIR/annotations_trainval2017.zip" \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    "$PY" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$COCO_DIR/annotations_trainval2017.zip" "$COCO_DIR"
    rm "$COCO_DIR/annotations_trainval2017.zip"
    echo "Annotations extracted to $COCO_DIR/annotations/"
else
    echo "Annotations already exist at $COCO_DIR/annotations/"
fi

# Validation images (needed as FID reference)
if [[ ! -d "$COCO_DIR/val2017" ]] || [[ $(ls "$COCO_DIR/val2017/" | wc -l) -lt 4000 ]]; then
    echo "Downloading COCO 2017 val images..."
    wget -q -O "$COCO_DIR/val2017.zip" \
        "http://images.cocodataset.org/zips/val2017.zip"
    "$PY" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$COCO_DIR/val2017.zip" "$COCO_DIR"
    rm "$COCO_DIR/val2017.zip"
    echo "Val images extracted to $COCO_DIR/val2017/"
else
    echo "Val images already exist at $COCO_DIR/val2017/"
fi

echo "COCO 2017 val ready at $COCO_DIR"
