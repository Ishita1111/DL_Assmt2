# DA6401 — Assignment 2: Visual Perception Pipeline

## Objective
Build a complete multi-task visual perception pipeline using PyTorch on the Oxford-IIIT Pet Dataset. The pipeline performs three tasks simultaneously in a single forward pass:
- **Classification** — Predict pet breed (37 classes)
- **Localization** — Predict bounding box around the pet's head
- **Segmentation** — Predict pixel-wise trimap mask (foreground / background / boundary)

---

## Project Structure

```
DL_Assmt2/
├── data/
│   └── pets_dataset.py         # Dataset class and dataloaders
├── models/
│   ├── __init__.py
│   ├── vgg11.py                # VGG11 backbone + ClassificationHead
│   ├── layers.py               # CustomDropout implementation
│   ├── classification.py       # VGG11Classifier wrapper
│   ├── localization.py         # VGG11Localizer + RegressionHead
│   ├── segmentation.py         # UNetDecoder + VGG11UNet
│   └── multitask.py            # Unified MultiTaskPerceptionModel
├── losses/
│   ├── __init__.py
│   └── iou_loss.py             # Custom IoU loss function
├── experiments/
│   ├── run_exp.py              # Single entry point for all W&B experiments
│   └── utils.py                # Shared helpers (backbones, hooks, train loops)
├── checkpoints/                # Saved model weights
├── train.py                    # Main training script
├── inference.py                # Inference script
├── requirements.txt
└── README.md
```

---

## Dataset
**Oxford-IIIT Pet Dataset** — 37 pet breeds with class labels, bounding boxes (head), and pixel-level trimaps.

Download: https://www.robots.ox.ac.uk/~vgg/data/pets/

Place the dataset at:
```
data/oxford-iiit-pet/
├── images/
└── annotations/
    ├── trainval.txt
    ├── test.txt
    ├── trimaps/
    └── xmls/
```

---

## Setup

```bash
pip install -r requirements.txt

---

## Training

```bash
python train.py
```

Trains the unified multi-task model for 20 epochs. Logs all metrics to W&B. Best weights are saved to `checkpoints/` based on composite score (F1 + mAP + Dice).

---

## Running Experiments (Section 2)

All Section 2 W&B experiments are in `experiments/run_exp.py`:

```bash
# Run a specific task
python experiments/run_exp.py --task 2_1
python experiments/run_exp.py --task 2_2
python experiments/run_exp.py --task 2_3
python experiments/run_exp.py --task 2_4
python experiments/run_exp.py --task 2_5
python experiments/run_exp.py --task 2_6
python experiments/run_exp.py --task 2_8

# Run all
python experiments/run_exp.py --all
```

| Task | Description |
|------|-------------|
| 2.1  | BatchNorm activation distribution analysis |
| 2.2  | Dropout generalization gap comparison |
| 2.3  | Transfer learning showdown (frozen / partial / full) |
| 2.4  | Feature map visualization (first vs last conv layer) |
| 2.5  | Object detection confidence & IoU table |
| 2.6  | Segmentation: Dice vs Pixel Accuracy |
| 2.7  | Final pipeline showcase on in-the-wild images |
| 2.8  | Meta-analysis and reflection |

---

## W&B Report
Public W&B report: [Wandb Report - ME22B134](https://wandb.ai/me22b134-indian-institute-of-technology-madras/da6401-assignment-2/reports/DA6401-Introduction-to-Deep-Learning--VmlldzoxNjQ4NDE3OQ?accessToken=j1o5zbfgg5m707z6dt96prt2lj506vg5gag41yv4jlbisa0q2zv0ashkoh7kndbx)

Github Repo Link: https://github.com/Ishita1111/DL_Assmt2
All runs are logged to project: `da6401-assignment-2`

---

## Model Weights
Weights are automatically downloaded from Google Drive when running experiments or inference. No manual download needed.
