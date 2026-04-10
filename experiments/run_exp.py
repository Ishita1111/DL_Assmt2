"""
experiments/run_experiments.py
Single entry point for all Section 2 W&B experiments.

Usage:
    python experiments/run_experiments.py --task 2_1
    python experiments/run_experiments.py --task 2_2
    python experiments/run_experiments.py --task 2_1 2_2
    python experiments/run_experiments.py --all
"""

import argparse
import sys
import os

# Make sure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb

from data.pets_dataset import get_dataloaders
from experiments.utils import (
    VGG11Backbone_WithBN, VGG11Backbone_NoBN,
    SimpleClassifier, register_hook,
    train_epoch, val_epoch, get_device
)

# ---------------------------------------------------------------------------
# Shared config — tweak here only
# ---------------------------------------------------------------------------
DATA_PATH  = "./data/oxford-iiit-pet"
BATCH_SIZE = 16
PROJECT    = "da6401-assignment-2"


# ===========================================================================
# TASK 2.1 — The Regularization Effect of BatchNorm
# ===========================================================================

def _run_single_bn_experiment(use_bn, lr, epochs, train_loader, val_loader,
                               device, fixed_batch):
    """Train one BN variant, log losses, return enc3 activations on fixed_batch."""
    label = "WithBN" if use_bn else "NoBN"

    run = wandb.init(
        project=PROJECT,
        name=f"task2_1_{label}",
        group="task2_1_batchnorm_analysis",
        config={"use_batchnorm": use_bn, "lr": lr, "epochs": epochs},
        reinit=True
    )

    backbone = VGG11Backbone_WithBN() if use_bn else VGG11Backbone_NoBN()
    model = SimpleClassifier(backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Hook on enc3's first Conv2d to capture activations
    handle, act_store = register_hook(backbone.enc3[0])

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)
        wandb.log({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss})
        print(f"  [{label}] Epoch {epoch}/{epochs}  train={tr_loss:.4f}  val={vl_loss:.4f}")

    # Capture activations on the fixed reference batch
    model.eval()
    with torch.no_grad():
        _ = model(fixed_batch.to(device))
    activations = act_store['output']   # [B, 256, H, W]

    handle.remove()
    wandb.finish()
    return activations


def task_2_1(epochs=10, lr=1e-4):
    print("\n" + "="*60)
    print("TASK 2.1 — BatchNorm Activation Distribution Analysis")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(root_dir=DATA_PATH, batch_size=BATCH_SIZE)

    # Fixed batch — same images for both models (fair comparison)
    fixed_batch, _, _, _ = next(iter(val_loader))

    print("\n--- Training WITH BatchNorm ---")
    act_bn = _run_single_bn_experiment(
        use_bn=True, lr=lr, epochs=epochs,
        train_loader=train_loader, val_loader=val_loader,
        device=device, fixed_batch=fixed_batch
    )

    print("\n--- Training WITHOUT BatchNorm ---")
    act_nobn = _run_single_bn_experiment(
        use_bn=False, lr=lr, epochs=epochs,
        train_loader=train_loader, val_loader=val_loader,
        device=device, fixed_batch=fixed_batch
    )

    # --- Plot & log comparison ---
    run = wandb.init(
        project=PROJECT,
        name="task2_1_distribution_comparison",
        group="task2_1_batchnorm_analysis",
        reinit=True
    )

    vals_bn   = act_bn.numpy().flatten()
    vals_nobn = act_nobn.numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("3rd Conv Layer Activation Distributions — WithBN vs NoBN (same input)",
                 fontsize=12)

    # Overlay histogram
    ax = axes[0]
    ax.hist(vals_nobn, bins=100, alpha=0.6, color='tomato',    label='Without BN', density=True)
    ax.hist(vals_bn,   bins=100, alpha=0.6, color='steelblue', label='With BN',    density=True)
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.set_title("Overlay Histogram")
    ax.legend()

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([vals_nobn, vals_bn], labels=['Without BN', 'With BN'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('tomato')
    bp['boxes'][1].set_facecolor('steelblue')
    ax2.set_title("Spread Comparison")
    ax2.set_ylabel("Activation Value")

    plt.tight_layout()
    wandb.log({"task2_1/activation_distribution": wandb.Image(fig)})
    plt.savefig("experiments/outputs/task2_1_activation_dist.png", dpi=150)
    plt.close()

    # Summary stats table
    table = wandb.Table(columns=["Model", "Mean", "Std", "Min", "Max", "Dead Neurons (%)"])
    for name, vals in [("Without BN", vals_nobn), ("With BN", vals_bn)]:
        dead = 100.0 * (vals <= 0).sum() / len(vals)
        table.add_data(name, round(float(vals.mean()), 4), round(float(vals.std()), 4),
                       round(float(vals.min()), 4), round(float(vals.max()), 4),
                       round(dead, 2))
    wandb.log({"task2_1/activation_stats": table})

    print("Task 2.1 complete. Plots logged to W&B.")
    wandb.finish()


# ===========================================================================
# TASK 2.2 — Internal Dynamics: Dropout Generalization Gap
# ===========================================================================

def _run_single_dropout_experiment(dropout_p, epochs, lr,
                                    train_loader, val_loader, device):
    """
    Train one dropout variant, log per-epoch train & val loss to W&B.
    dropout_p = None  → No Dropout
    dropout_p = 0.2   → Custom Dropout p=0.2
    dropout_p = 0.5   → Custom Dropout p=0.5
    Returns (train_losses, val_losses) lists for local overlay plot.
    """
    if dropout_p is None:
        label = "no_dropout"
        p_val = 0.0
    else:
        label = f"dropout_p{str(dropout_p).replace('.','')}"
        p_val = dropout_p

    run = wandb.init(
        project=PROJECT,
        name=f"task2_2_{label}",
        group="task2_2_internal_dynamics",
        config={"dropout_p": p_val, "lr": lr, "epochs": epochs},
        reinit=True
    )

    # Build model: WithBN backbone + head with chosen dropout
    backbone = VGG11Backbone_WithBN()

    if dropout_p is None:
        # Replace CustomDropout with Identity so no dropout is applied
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Identity(),
            nn.Linear(4096, 4096),         nn.ReLU(True), nn.Identity(),
            nn.Linear(4096, 37)
        )
    else:
        from models.layers import CustomDropout
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), CustomDropout(dropout_p),
            nn.Linear(4096, 4096),         nn.ReLU(True), CustomDropout(dropout_p),
            nn.Linear(4096, 37)
        )

    model = nn.Sequential(backbone, head).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        tr_total = 0.0
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            tr_total += loss.item()
        tr_loss = tr_total / len(train_loader)

        # --- val ---
        model.eval()
        vl_total = 0.0
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                vl_total += criterion(model(images), labels).item()
        vl_loss = vl_total / len(val_loader)

        gap = tr_loss - vl_loss

        wandb.log({
            "epoch":        epoch,
            "train_loss":   tr_loss,
            "val_loss":     vl_loss,
            "generalization_gap": gap,   # key metric for the report
        })

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        print(f"  [{label}] Epoch {epoch}/{epochs}  "
              f"train={tr_loss:.4f}  val={vl_loss:.4f}  gap={gap:.4f}")

    wandb.finish()
    return train_losses, val_losses


def task_2_2(epochs=10, lr=1e-4):
    print("\n" + "="*60)
    print("TASK 2.2 — Internal Dynamics: Dropout Generalization Gap")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(root_dir=DATA_PATH, batch_size=BATCH_SIZE)

    results = {}   # label -> (train_losses, val_losses)

    conditions = [
        (None,  "No Dropout"),
        (0.2,   "Dropout p=0.2"),
        (0.5,   "Dropout p=0.5"),
    ]

    for p, name in conditions:
        print(f"\n--- {name} ---")
        tr_losses, vl_losses = _run_single_dropout_experiment(
            dropout_p=p, epochs=epochs, lr=lr,
            train_loader=train_loader, val_loader=val_loader, device=device
        )
        results[name] = (tr_losses, vl_losses)

    # -------------------------------------------------------------------
    # Local overlay plot (also logged to W&B in a summary run)
    # -------------------------------------------------------------------
    run = wandb.init(
        project=PROJECT,
        name="task2_2_overlay_comparison",
        group="task2_2_internal_dynamics",
        reinit=True
    )

    colors = {
        "No Dropout":    ("firebrick",   "salmon"),
        "Dropout p=0.2": ("darkorange",  "moccasin"),
        "Dropout p=0.5": ("steelblue",   "lightskyblue"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Task 2.2 — Train vs Val Loss: Dropout Comparison", fontsize=13)

    epochs_range = range(1, epochs + 1)

    # Left: Training loss overlay
    ax1 = axes[0]
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Right: Validation loss overlay
    ax2 = axes[1]
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    for name, (tr, vl) in results.items():
        tc, vc = colors[name]
        ax1.plot(epochs_range, tr, color=tc, linewidth=2,   label=name)
        ax2.plot(epochs_range, vl, color=tc, linewidth=2,   label=name)
        # Shade the generalization gap on a combined axes
        ax1.fill_between(epochs_range, tr, vl,
                         alpha=0.08, color=tc)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    wandb.log({"task2_2/loss_overlay": wandb.Image(fig)})
    plt.savefig("experiments/outputs/task2_2_dropout_comparison.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------
    # Generalization gap summary table
    # -------------------------------------------------------------------
    table = wandb.Table(
        columns=["Condition", "Final Train Loss", "Final Val Loss",
                 "Final Gap", "Avg Gap (all epochs)"]
    )
    for name, (tr, vl) in results.items():
        gaps = [t - v for t, v in zip(tr, vl)]
        table.add_data(
            name,
            round(tr[-1], 4),
            round(vl[-1], 4),
            round(gaps[-1], 4),
            round(float(np.mean(gaps)), 4)
        )
    wandb.log({"task2_2/generalization_gap_summary": table})

    print("\nTask 2.2 complete. Plots logged to W&B.")
    wandb.finish()


# ===========================================================================
# TASK 2.3  — (placeholder)
# ===========================================================================

def task_2_3():
    raise NotImplementedError("Task 2.3 not yet implemented.")


# ===========================================================================
# TASK 2.4 — Inside the Black Box: Feature Maps
# ===========================================================================

def task_2_4(checkpoint_path="checkpoints/classifier.pth",
             dog_image_path=None):
    """
    Loads the trained VGG11Classifier, passes a single dog image through it,
    extracts and visualises feature maps from:
      - enc1[0]  : first Conv2d  (64 channels, 224x224)
      - enc5[-3] : last Conv2d before pool5  (512 channels, 14x14)
    Logs everything to W&B.

    Args:
        checkpoint_path : path to classifier.pth saved by train.py
        dog_image_path  : path to a dog image file; if None uses first
                          dog image found in the val split.
    """
    print("\n" + "="*60)
    print("TASK 2.4 — Inside the Black Box: Feature Maps")
    print("="*60)

    import torchvision.transforms as T
    from PIL import Image
    from models.vgg11 import VGG11Backbone, ClassificationHead

    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load trained classifier
    # ------------------------------------------------------------------
    backbone = VGG11Backbone()
    head     = ClassificationHead(num_classes=37)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone.load_state_dict(ckpt['backbone'])
    head.load_state_dict(ckpt['classifier_head'])

    backbone = backbone.to(device).eval()
    head     = head.to(device).eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # ------------------------------------------------------------------
    # 2. Get a single dog image
    # ------------------------------------------------------------------
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    if dog_image_path is not None:
        img_pil = Image.open(dog_image_path).convert("RGB")
    else:
        # Pull first image from val loader and use it
        _, val_loader = get_dataloaders(root_dir=DATA_PATH, batch_size=1)
        raw_img, _, _, _ = next(iter(val_loader))
        # Denormalise for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_display = (raw_img[0] * std + mean).clamp(0, 1)
        input_tensor = raw_img.to(device)
        img_pil = None   # already have tensor

    if img_pil is not None:
        input_tensor  = transform(img_pil).unsqueeze(0).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_display = (input_tensor[0].cpu() * std + mean).clamp(0, 1)

    # ------------------------------------------------------------------
    # 3. Register hooks on enc1[0] and enc5[-3]
    #    enc5 = Sequential(Conv, BN, ReLU, Conv, BN, ReLU)
    #    indices:           0    1   2     3    4   5
    #    last Conv before pool5 = enc5[3]
    # ------------------------------------------------------------------
    handle1, store1 = register_hook(backbone.enc1[0])   # first conv
    handle5, store5 = register_hook(backbone.enc5[3])   # last conv before pool5

    with torch.no_grad():
        _ = backbone(input_tensor)

    handle1.remove()
    handle5.remove()

    fmap_first = store1['output'][0]   # [64,  224, 224]
    fmap_last  = store5['output'][0]   # [512,  14,  14]

    # ------------------------------------------------------------------
    # 4. Plot helper: grid of N feature map channels
    # ------------------------------------------------------------------
    def plot_feature_maps(fmap, title, n_show=32, save_name=None):
        """Plots up to n_show channels from a feature map tensor [C, H, W]."""
        n_show  = min(n_show, fmap.shape[0])
        cols    = 8
        rows    = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 1.8, rows * 1.8))
        fig.suptitle(title, fontsize=11)
        axes = axes.flatten()
        for i in range(n_show):
            ch = fmap[i].numpy()
            axes[i].imshow(ch, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f"ch {i}", fontsize=6)
        for j in range(n_show, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        if save_name:
            plt.savefig(f"experiments/outputs/{save_name}", dpi=120)
        return fig

    # ------------------------------------------------------------------
    # 5. W&B logging
    # ------------------------------------------------------------------
    run = wandb.init(
        project=PROJECT,
        name="task2_4_feature_maps",
        group="task2_4_black_box",
        config={"checkpoint": checkpoint_path},
        reinit=True
    )

    # Log the input image
    img_np = (img_display.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    wandb.log({"task2_4/input_image": wandb.Image(img_np, caption="Input dog image")})

    # First conv feature maps (64 channels)
    fig_first = plot_feature_maps(
        fmap_first,
        title=f"enc1[0] — First Conv Layer  (64 channels, {fmap_first.shape[1]}x{fmap_first.shape[2]})\n"
              "Low-level: edges, colours, textures",
        n_show=64,
        save_name="task2_4_first_conv_fmaps.png"
    )
    wandb.log({"task2_4/first_conv_feature_maps": wandb.Image(fig_first)})
    plt.close(fig_first)

    # Last conv feature maps (show 64 of 512 channels)
    fig_last = plot_feature_maps(
        fmap_last,
        title=f"enc5[3] — Last Conv Layer  (512 channels, {fmap_last.shape[1]}x{fmap_last.shape[2]})\n"
              "High-level: semantic parts (snout, ears, fur patterns)",
        n_show=64,
        save_name="task2_4_last_conv_fmaps.png"
    )
    wandb.log({"task2_4/last_conv_feature_maps": wandb.Image(fig_last)})
    plt.close(fig_last)

    # Side-by-side mean activation map comparison
    fig_cmp, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig_cmp.suptitle("Task 2.4 — Mean Activation Map Comparison", fontsize=12)

    axes[0].imshow(img_display.permute(1, 2, 0).numpy())
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    mean_first = fmap_first.mean(0).numpy()
    im1 = axes[1].imshow(mean_first, cmap='hot')
    axes[1].set_title(f"Mean of enc1 maps\n(first conv, {fmap_first.shape[0]} ch)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    mean_last = fmap_last.mean(0).numpy()
    im2 = axes[2].imshow(mean_last, cmap='hot')
    axes[2].set_title(f"Mean of enc5 maps\n(last conv, {fmap_last.shape[0]} ch)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    wandb.log({"task2_4/mean_activation_comparison": wandb.Image(fig_cmp)})
    plt.savefig("experiments/outputs/task2_4_mean_activation_cmp.png", dpi=150)
    plt.close(fig_cmp)

    # Stats table
    table = wandb.Table(columns=["Layer", "Channels", "Spatial Size",
                                  "Mean Activation", "Std", "Dead (%)"])
    for lname, fm in [("enc1[0] — First Conv", fmap_first),
                      ("enc5[3] — Last Conv",  fmap_last)]:
        v = fm.numpy().flatten()
        dead = 100.0 * (v <= 0).sum() / len(v)
        table.add_data(lname, fm.shape[0],
                       f"{fm.shape[1]}x{fm.shape[2]}",
                       round(float(v.mean()), 4),
                       round(float(v.std()),  4),
                       round(dead, 2))
    wandb.log({"task2_4/feature_map_stats": table})

    print("Task 2.4 complete. Feature maps logged to W&B.")
    wandb.finish()


# ===========================================================================
# TASK 2.5  — (placeholder)
# ===========================================================================

def task_2_5():
    raise NotImplementedError("Task 2.5 not yet implemented.")


# ===========================================================================
# TASK 2.6  — (placeholder)
# ===========================================================================

def task_2_6():
    raise NotImplementedError("Task 2.6 not yet implemented.")


# ===========================================================================
# TASK 2.7  — (placeholder)
# ===========================================================================

def task_2_7():
    raise NotImplementedError("Task 2.7 not yet implemented.")


# ===========================================================================
# TASK 2.8  — (placeholder)
# ===========================================================================

def task_2_8():
    raise NotImplementedError("Task 2.8 not yet implemented.")


# ===========================================================================
# CLI
# ===========================================================================

TASK_MAP = {
    "2_1": task_2_1,
    "2_2": task_2_2,
    "2_3": task_2_3,
    "2_4": task_2_4,
    "2_5": task_2_5,
    "2_6": task_2_6,
    "2_7": task_2_7,
    "2_8": task_2_8,
}

if __name__ == "__main__":
    # Create output dir for saved plots
    os.makedirs("experiments/outputs", exist_ok=True)

    parser = argparse.ArgumentParser(description="Run Section 2 W&B Experiments")
    parser.add_argument(
        "--task", nargs="+",
        choices=list(TASK_MAP.keys()),
        help="Task(s) to run, e.g. --task 2_1 2_2"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tasks sequentially"
    )
    args = parser.parse_args()

    if args.all:
        tasks_to_run = list(TASK_MAP.keys())
    elif args.task:
        tasks_to_run = args.task
    else:
        parser.print_help()
        sys.exit(1)

    for t in tasks_to_run:
        print(f"\n>>> Running task {t}")
        try:
            TASK_MAP[t]()
        except NotImplementedError:
            print(f"  Task {t} not yet implemented — skipping.")