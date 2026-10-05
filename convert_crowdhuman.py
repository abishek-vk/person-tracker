from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _find_annotation_files(source: Path, explicit: Path | None) -> list[Path]:
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(f"CrowdHuman annotation file not found: {explicit}")
        return [explicit]
    candidates = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".odgt", ".json"})
    if not candidates:
        raise RuntimeError(
            "No CrowdHuman annotations were found. CrowdHuman conversion requires the official .odgt files; "
            "image-only cropped data cannot be converted to correct full-image bounding boxes."
        )
    return candidates


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".odgt":
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("images", data.get("annotations", []))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return data


def _image_index(source: Path) -> dict[str, Path]:
    return {path.stem: path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES}


def _convert_split(records: list[dict[str, Any]], image_index: dict[str, Path], output: Path, split: str) -> tuple[int, int]:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_count = box_count = 0
    for record in records:
        image_id = str(record.get("ID", record.get("id", record.get("name", ""))))
        image = image_index.get(Path(image_id).stem)
        if image is None:
            continue
        with Image.open(image) as opened:
            width, height = opened.size
        labels = []
        for gtbox in record.get("gtboxes", record.get("boxes", [])):
            if gtbox.get("tag", "person") != "person":
                continue
            box = gtbox.get("fbox", gtbox.get("bbox"))
            if not box or len(box) != 4:
                continue
            x, y, box_width, box_height = map(float, box)
            if box_width <= 0 or box_height <= 0:
                continue
            labels.append(
                f"0 {(x + box_width / 2) / width:.6f} {(y + box_height / 2) / height:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
        shutil.copy2(image, image_dir / image.name)
        (label_dir / f"{image.stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        image_count += 1
        box_count += len(labels)
    return image_count, box_count


def convert(source: Path, output: Path, annotations: Path | None = None) -> Path:
    files = _find_annotation_files(source, annotations)
    image_index = _image_index(source)
    output.mkdir(parents=True, exist_ok=True)
    split_records = {"train": _records(files[0]), "val": _records(files[1]) if len(files) > 1 else []}
    totals = {split: _convert_split(records, image_index, output, split) for split, records in split_records.items()}
    if not totals["train"][0] or not totals["train"][1]:
        raise RuntimeError("CrowdHuman annotations were found, but no train images/person boxes matched them.")
    if not totals["val"][0]:
        raise RuntimeError("CrowdHuman validation annotations did not match any local images.")
    yaml_path = output / "crowdhuman.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [person]\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official CrowdHuman .odgt annotations to YOLO person labels.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.output, args.annotations))


if __name__ == "__main__":
    main()