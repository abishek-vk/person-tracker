from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_yolo(root: Path, yaml_name: str) -> None:
    if not (root / yaml_name).is_file():
        raise RuntimeError(f"Missing dataset YAML: {root / yaml_name}")
    image_count = label_count = 0
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise RuntimeError(f"No {split} images found in {image_dir}")
        for image in images:
            with Image.open(image):
                pass
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise RuntimeError(f"Missing label for {image}")
            lines = label.read_text(encoding="utf-8").splitlines()
            for line in lines:
                values = line.split()
                if len(values) != 5 or values[0] != "0" or any(not 0 <= float(value) <= 1 for value in values[1:]):
                    raise RuntimeError(f"Invalid person-only YOLO label: {label}")
            image_count += 1
            label_count += len(lines)
    if not label_count:
        raise RuntimeError(f"No person boxes found under {root}")
    print(f"Valid YOLO dataset: {root} ({image_count} images, {label_count} person boxes)")


def validate_csv_dataset(root: Path) -> None:
    total = 0
    for split in ("train", "valid", "test"):
        split_root = root / split
        csv_files = list(split_root.rglob("_annotations.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(f"Expected one Roboflow annotation CSV under {split_root}")
        csv_path = csv_files[0]
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("class", "").strip().lower() != "person":
                raise RuntimeError(f"Non-person class found in {csv_path}: {row.get('class')}")
            if not (float(row["xmin"]) < float(row["xmax"]) and float(row["ymin"]) < float(row["ymax"])):
                raise RuntimeError(f"Invalid box in {csv_path}: {row}")
        total += len(rows)
    if not total:
        raise RuntimeError(f"No person boxes found under {root}")
    print(f"Valid person CSV dataset: {root} ({total} boxes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate person-only detection datasets.")
    parser.add_argument("--yolo", type=Path)
    parser.add_argument("--csv-root", type=Path)
    parser.add_argument("--yaml", default="data.yaml")
    args = parser.parse_args()
    if bool(args.yolo) == bool(args.csv_root):
        parser.error("provide exactly one of --yolo or --csv-root")
    validate_yolo(args.yolo, args.yaml) if args.yolo else validate_csv_dataset(args.csv_root)


if __name__ == "__main__":
    main()