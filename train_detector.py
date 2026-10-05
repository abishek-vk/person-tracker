from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from convert_crowdhuman import convert as convert_crowdhuman
from validate_dataset import validate_csv_dataset, validate_yolo

ROOT = Path(__file__).resolve().parent
CROWD_SOURCE = ROOT / "datasets" / "crowd human"
PEOPLE_SOURCE = ROOT / "datasets" / "person detection"
PREPARED = ROOT / "prepared_datasets"
CROWD_YOLO = PREPARED / "crowdhuman"
PEOPLE_YOLO = PREPARED / "people"
MODEL = ROOT / "yolo26m.pt"


def device_and_batch(fast: bool = False) -> tuple[str, int]:
    if not torch.cuda.is_available():
        print("CUDA is unavailable in this Python environment; using CPU with batch=1.")
        return "cpu", 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if fast and vram_gb >= 6:
        batch = 8 if vram_gb < 10 else 16
    else:
        batch = 16 if vram_gb >= 16 else 8 if vram_gb >= 10 else 4 if vram_gb >= 6 else 2
    print(f"Using CUDA GPU ({vram_gb:.1f} GB VRAM) with batch={batch}.")
    return "0", batch


def prepare_people_dataset() -> Path:
    yaml_path = PEOPLE_YOLO / "data.yaml"
    if PEOPLE_YOLO.exists() and yaml_path.is_file():
        yaml_path.write_text(
            f"path: {PEOPLE_YOLO.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: [person]\n",
            encoding="utf-8",
        )
        return yaml_path
    validate_csv_dataset(PEOPLE_SOURCE)
    for source_split, target_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        source = PEOPLE_SOURCE / source_split
        image_dir = PEOPLE_YOLO / "images" / target_split
        label_dir = PEOPLE_YOLO / "labels" / target_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        annotations: dict[str, list[str]] = {}
        csv_files = list(source.rglob("_annotations.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(f"Expected one annotation CSV under {source}")
        image_index = {path.name: path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        with csv_files[0].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                filename = row["filename"]
                annotations.setdefault(filename, []).append(
                    f"0 {(float(row['xmin']) + float(row['xmax'])) / (2 * float(row['width'])):.6f} "
                    f"{(float(row['ymin']) + float(row['ymax'])) / (2 * float(row['height'])):.6f} "
                    f"{(float(row['xmax']) - float(row['xmin'])) / float(row['width']):.6f} "
                    f"{(float(row['ymax']) - float(row['ymin'])) / float(row['height']):.6f}"
                )
        for filename, labels in annotations.items():
            if filename not in image_index:
                raise RuntimeError(f"Image listed in annotations was not found: {filename}")
            shutil.copy2(image_index[filename], image_dir / filename)
            (label_dir / f"{Path(filename).stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    yaml_path.write_text(
        f"path: {PEOPLE_YOLO.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: [person]\n",
        encoding="utf-8",
    )
    validate_yolo(PEOPLE_YOLO, "data.yaml")
    return PEOPLE_YOLO / "data.yaml"


def train_stage(model_path: Path, data_yaml: Path, project: Path, name: str, epochs: int, device: str, batch: int, imgsz: int = 640, patience: int = 10) -> Path:
    model = YOLO(str(model_path))
    model.train(data=str(data_yaml), epochs=epochs, patience=patience, imgsz=imgsz, amp=True, cache=False, workers=0, batch=batch, device=device, project=str(project), name=name, exist_ok=True, single_cls=True)
    best = project / name / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished without producing {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26m sequential person detector training pipeline.")
    parser.add_argument("--pipeline", action="store_true", required=True)
    parser.add_argument("--test", action="store_true", help="Use 5 epochs for each stage.")
    parser.add_argument("--people-only", action="store_true", help="Train People Detection directly from YOLO26m pretrained weights.")
    parser.add_argument("--fast", action="store_true", help="Use the under-8-hour People Detection profile: 512px, larger batch, 15 epochs.")
    args = parser.parse_args()
    if not MODEL.is_file():
        raise FileNotFoundError(f"YOLO26m weights not found: {MODEL}")
    device, batch = device_and_batch(args.fast)
    project = ROOT / "runs" / "person_detection"
    if args.people_only:
        people_yaml = prepare_people_dataset()
        fast_imgsz = 512 if args.fast else 640
        fast_epochs = 5 if args.test else 15 if args.fast else 40
        fast_patience = 5 if args.fast else 10
        final_best = train_stage(MODEL, people_yaml, project, "yolo26m_people", fast_epochs, device, batch, fast_imgsz, fast_patience)
        print(f"Final model: {final_best}")
        return
    crowd_yaml = convert_crowdhuman(CROWD_SOURCE, CROWD_YOLO)
    validate_yolo(CROWD_YOLO, "crowdhuman.yaml")
    people_yaml = prepare_people_dataset()
    crowd_best = train_stage(MODEL, crowd_yaml, project, "yolo26m_crowdhuman", 5 if args.test else 60, device, batch)
    print(f"CrowdHuman complete: {crowd_best}")
    final_best = train_stage(crowd_best, people_yaml, project, "yolo26m_people", 5 if args.test else 40, device, batch)
    print(f"Final model: {final_best}")


if __name__ == "__main__":
    main()