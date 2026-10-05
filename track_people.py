from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person detection and tracking with YOLO26m and BoT-SORT"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="outputs/tracked.mp4")
    parser.add_argument("--crops", default="outputs/person_crops")
    parser.add_argument("--metadata", default="outputs/tracks.csv")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    crops_path = Path(args.crops)
    metadata_path = Path(args.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crops_path.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    video = cv2.VideoCapture(args.input)
    if not video.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            metadata_writer = csv.writer(csv_file)
            metadata_writer.writerow(
                [
                    "frame",
                    "track_id",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "crop_path",
                ]
            )

            frame_number = 0
            while True:
                success, frame = video.read()
                if not success:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=[0],
                    conf=args.confidence,
                    verbose=False,
                )
                result = results[0]
                writer.write(result.plot())

                if result.boxes is None or result.boxes.id is None:
                    frame_number += 1
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for box, track_id, confidence in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = [int(value) for value in box]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_directory = crops_path / f"person_{track_id}"
                    person_directory.mkdir(parents=True, exist_ok=True)
                    crop_path = person_directory / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])

                    metadata_writer.writerow(
                        [
                            frame_number,
                            track_id,
                            f"{confidence:.4f}",
                            x1,
                            y1,
                            x2,
                            y2,
                            str(crop_path),
                        ]
                    )

                frame_number += 1
    finally:
        video.release()
        writer.release()

    print(f"Tracked video: {output_path}")
    print(f"Person crops:  {crops_path}")
    print(f"Metadata CSV:  {metadata_path}")


if __name__ == "__main__":
    main()
