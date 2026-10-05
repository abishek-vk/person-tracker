# YOLO26m Sequential Person Detector

This project trains YOLO26m only in one automatic pipeline: YOLO26m pretrained -> CrowdHuman -> CrowdHuman best.pt -> People Detection -> final best.pt.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python train_detector.py --pipeline
```

Test mode trains five epochs on each dataset:

```text
```powershell
python train_detector.py --pipeline --test
```

To start directly with the available People Detection dataset:

```powershell
python train_detector.py --pipeline --people-only
```

For the RTX 5060 8 GB profile targeting completion within about eight hours:

```powershell
python train_detector.py --pipeline --people-only --fast
```

This keeps YOLO26m and AMP, uses 512px images, batch 8, and 15 epochs with patience 5. The standard mode remains 640px and 40 epochs.

This direct mode starts from `yolo26m.pt` and writes `runs/person_detection/yolo26m_people/weights/best.pt`; it does not include CrowdHuman pretraining.
```

The pipeline uses 640px images, AMP, disabled caching, two workers, patience 10, and a batch selected from available GPU VRAM. Stage 2 loads `runs/person_detection/yolo26m_crowdhuman/weights/best.pt` and writes the final model to `runs/person_detection/yolo26m_people/weights/best.pt`.

## Output

The CrowdHuman converter requires official `.odgt` annotations and matching full images. The current local CrowdHuman copy contains only cropped JPGs and no annotations, so it correctly stops with an actionable error instead of inventing boxes. The People Detection data is read from its Roboflow CSV files and converted automatically to class `0 = person`.

This project does not implement BoT-SORT, Re-ID, persistent IDs, or video inference.
