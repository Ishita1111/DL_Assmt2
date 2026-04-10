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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

def task_2_4(checkpoint_path=os.path.join(_PROJECT_ROOT, "checkpoints", "classifier.pth"),
             dog_image_path=None):
    
    # Download weights if not already present
    import gdown
    os.makedirs(os.path.join(_PROJECT_ROOT, "checkpoints"), exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        print("Downloading classifier.pth from Google Drive...")
        gdown.download(id="1kXjDJLjXMzz4HIyiUpNIljOr2Gko48_a", output=checkpoint_path, quiet=False)
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
# TASK 2.5 — Object Detection: Confidence & IoU
# ===========================================================================

def task_2_5(n_images=15):
    print("\n" + "="*60)
    print("TASK 2.5 — Object Detection: Confidence & IoU")
    print("="*60)

    import gdown
    from PIL import Image, ImageDraw
    from models.vgg11 import VGG11Backbone
    from models.localization import RegressionHead

    device = get_device()
    print(f"Device: {device}")

    # --- Download & load weights ---
    ckpt_dir  = os.path.join(_PROJECT_ROOT, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "localizer.pth")
    cls_path  = os.path.join(ckpt_dir, "classifier.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(cls_path):
        print("Downloading classifier.pth...")
        gdown.download(id="1kXjDJLjXMzz4HIyiUpNIljOr2Gko48_a", output=cls_path, quiet=False)
    if not os.path.exists(ckpt_path):
        print("Downloading localizer.pth...")
        gdown.download(id="10g0EduM3-vnuTBrvQh33LIhRUSVzcgug", output=ckpt_path, quiet=False)

    backbone = VGG11Backbone()
    backbone.load_state_dict(torch.load(cls_path, map_location="cpu")['backbone'])
    locator = RegressionHead()
    locator.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    backbone = backbone.to(device).eval()
    locator  = locator.to(device).eval()

    _, val_loader = get_dataloaders(root_dir=DATA_PATH, batch_size=1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def compute_iou(pred_px, gt_px):
        def to_xyxy(b):
            return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
        px1,py1,px2,py2 = to_xyxy(pred_px)
        gx1,gy1,gx2,gy2 = to_xyxy(gt_px)
        inter = max(0, min(px2,gx2)-max(px1,gx1)) * max(0, min(py2,gy2)-max(py1,gy1))
        union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
        return inter / (union + 1e-6)

    def draw_boxes(img_t, pred_px, gt_px):
        img_pil = Image.fromarray(
            ((img_t * std + mean).clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
        )
        draw = ImageDraw.Draw(img_pil)
        W, H = img_pil.size
        def box(b): return [max(0,int(b[0]-b[2]/2)), max(0,int(b[1]-b[3]/2)),
                             min(W,int(b[0]+b[2]/2)), min(H,int(b[1]+b[3]/2))]
        draw.rectangle(box(gt_px),   outline="green", width=3)
        draw.rectangle(box(pred_px), outline="red",   width=3)
        return img_pil

    wandb.init(project=PROJECT, name="task2_5_detection_iou",
               group="task2_5_object_detection",
               config={"n_images": n_images}, reinit=True)

    table = wandb.Table(columns=["Image", "IoU", "Confidence",
                                  "Pred [cx,cy,w,h]", "GT [cx,cy,w,h]", "Failure?"])
    all_ious, count = [], 0

    with torch.no_grad():
        for img_t, _, bbox_gt, _ in val_loader:
            if count >= n_images: break
            bottleneck, _ = backbone(img_t.to(device))
            pred_norm = locator(bottleneck)[0].cpu()   # [4] normalized

            H = W = 224
            pred_px = [pred_norm[0]*W, pred_norm[1]*H, pred_norm[2]*W, pred_norm[3]*H]
            gt_norm = bbox_gt[0].cpu().tolist()
            gt_px   = [gt_norm[0]*W,   gt_norm[1]*H,   gt_norm[2]*W,   gt_norm[3]*H]

            iou = compute_iou(pred_px, gt_px)
            l1  = float(torch.abs(pred_norm - bbox_gt[0].cpu()).mean())
            confidence = round(1.0 / (1.0 + l1), 4)
            is_failure = confidence > 0.6 and iou < 0.3

            img_drawn = draw_boxes(img_t[0].cpu(), pred_px, gt_px)
            table.add_data(
                wandb.Image(img_drawn),
                round(float(iou), 4), confidence,
                str([round(x,1) for x in pred_px]),
                str([round(x,4) for x in gt_norm]),
                "YES ⚠️" if is_failure else "no"
            )
            all_ious.append(iou)
            count += 1

    wandb.log({"task2_5/detection_results_table": table})

    ious_arr = np.array(all_ious)
    summary  = wandb.Table(columns=["Metric", "Value"])
    summary.add_data("Mean IoU",            round(float(ious_arr.mean()), 4))
    summary.add_data("Median IoU",          round(float(np.median(ious_arr)), 4))
    summary.add_data("mAP@50",              f"{(ious_arr>0.5).sum()}/{len(ious_arr)}")
    summary.add_data("Worst IoU",           round(float(ious_arr.min()), 4))
    summary.add_data("Best IoU",            round(float(ious_arr.max()), 4))
    wandb.log({"task2_5/iou_summary": summary})

    print(f"Task 2.5 complete. {count} images logged to W&B.")
    wandb.finish()


# ===========================================================================
# TASK 2.6 — Segmentation Evaluation: Dice vs Pixel Accuracy
# ===========================================================================

def task_2_6(n_images=5):
    print("\n" + "="*60)
    print("TASK 2.6 — Segmentation: Dice vs Pixel Accuracy")
    print("="*60)

    import gdown
    from models.vgg11 import VGG11Backbone
    from models.segmentation import UNetDecoder

    device = get_device()
    print(f"Device: {device}")

    # --- Download & load weights ---
    ckpt_dir   = os.path.join(_PROJECT_ROOT, "checkpoints")
    cls_path   = os.path.join(ckpt_dir, "classifier.pth")
    unet_path  = os.path.join(ckpt_dir, "unet.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(cls_path):
        print("Downloading classifier.pth...")
        gdown.download(id="1kXjDJLjXMzz4HIyiUpNIljOr2Gko48_a", output=cls_path, quiet=False)
    if not os.path.exists(unet_path):
        print("Downloading unet.pth...")
        gdown.download(id="1DIBo-YerVcDUluFtqaMeKzR8f74Btcy4", output=unet_path, quiet=False)

    backbone  = VGG11Backbone()
    backbone.load_state_dict(torch.load(cls_path, map_location="cpu")['backbone'])
    segmenter = UNetDecoder(num_classes=3)
    segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))
    backbone  = backbone.to(device).eval()
    segmenter = segmenter.to(device).eval()

    _, val_loader = get_dataloaders(root_dir=DATA_PATH, batch_size=1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # Trimap class colours for visualisation: 0=bg, 1=fg, 2=boundary
    PALETTE = np.array([[0,0,0], [255,255,255], [128,128,128]], dtype=np.uint8)

    def mask_to_rgb(mask_np):
        """Convert [H,W] class index mask to [H,W,3] RGB using palette."""
        return PALETTE[mask_np]

    def compute_dice(pred, target, num_classes=3):
        """Macro Dice over all classes. pred/target: [H,W] numpy arrays."""
        smooth = 1e-6
        scores = []
        for c in range(num_classes):
            p = (pred == c).astype(float)
            t = (target == c).astype(float)
            inter = (p * t).sum()
            scores.append((2 * inter + smooth) / (p.sum() + t.sum() + smooth))
        return float(np.mean(scores))

    def compute_pixel_acc(pred, target):
        return float((pred == target).sum() / target.size)

    # ------------------------------------------------------------------
    # Run inference and collect per-image metrics
    # ------------------------------------------------------------------
    wandb.init(project=PROJECT, name="task2_6_segmentation_dice_vs_pixacc",
               group="task2_6_segmentation_evaluation",
               config={"n_images": n_images}, reinit=True)

    # Table: one row per image
    img_table = wandb.Table(columns=[
        "Original Image", "Ground Truth Trimap", "Predicted Trimap",
        "Pixel Accuracy", "Dice Score", "Gap (PA - Dice)"
    ])

    # Metric accumulators over full val set for curve logging
    all_pixel_acc, all_dice = [], []
    count = 0

    with torch.no_grad():
        for img_t, _, _, seg_gt in val_loader:
            img_t  = img_t.to(device)
            seg_gt = seg_gt.to(device)                      # [1, H, W]

            bottleneck, skip = backbone(img_t)
            seg_logits = segmenter(bottleneck, skip)        # [1, 3, H, W]
            pred_mask  = seg_logits.argmax(dim=1)           # [1, H, W]

            pred_np = pred_mask[0].cpu().numpy()            # [H, W]
            gt_np   = seg_gt[0].cpu().numpy()               # [H, W]

            dice     = compute_dice(pred_np, gt_np)
            pix_acc  = compute_pixel_acc(pred_np, gt_np)
            all_dice.append(dice)
            all_pixel_acc.append(pix_acc)

            # Log first n_images visually
            if count < n_images:
                # Original image (denormalised)
                orig_np = ((img_t[0].cpu() * std + mean)
                           .clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)

                gt_rgb   = mask_to_rgb(gt_np)
                pred_rgb = mask_to_rgb(pred_np)

                img_table.add_data(
                    wandb.Image(orig_np,  caption="Original"),
                    wandb.Image(gt_rgb,   caption="GT Trimap"),
                    wandb.Image(pred_rgb, caption="Predicted Trimap"),
                    round(pix_acc, 4),
                    round(dice, 4),
                    round(pix_acc - dice, 4)   # gap — key insight for report
                )
                count += 1

    wandb.log({"task2_6/sample_segmentation_table": img_table})

    # ------------------------------------------------------------------
    # Log Pixel Accuracy vs Dice Score scatter + bar comparison
    # ------------------------------------------------------------------
    all_pixel_acc = np.array(all_pixel_acc)
    all_dice      = np.array(all_dice)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 2.6 — Pixel Accuracy vs Dice Score (full val set)", fontsize=12)

    # Scatter: each point = one image
    ax1 = axes[0]
    ax1.scatter(all_pixel_acc, all_dice, alpha=0.5, color='steelblue', s=15)
    ax1.plot([0,1],[0,1], 'r--', linewidth=1, label='PA = Dice line')
    ax1.set_xlabel("Pixel Accuracy")
    ax1.set_ylabel("Dice Score")
    ax1.set_title("PA vs Dice per image\n(points above red = PA inflated)")
    ax1.legend()

    # Bar: mean comparison
    ax2 = axes[1]
    means = [all_pixel_acc.mean(), all_dice.mean()]
    bars  = ax2.bar(["Pixel Accuracy", "Dice Score"], means,
                    color=['coral', 'steelblue'], width=0.4)
    ax2.set_ylim(0, 1)
    ax2.set_title("Mean over Val Set")
    ax2.set_ylabel("Score")
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha='center', fontsize=11)

    plt.tight_layout()
    wandb.log({"task2_6/pixel_acc_vs_dice_plot": wandb.Image(fig)})
    plt.savefig("experiments/outputs/task2_6_dice_vs_pixacc.png", dpi=150)
    plt.close(fig)

    # Summary stats table
    summary = wandb.Table(columns=["Metric", "Mean", "Std", "Min", "Max"])
    for name, arr in [("Pixel Accuracy", all_pixel_acc), ("Dice Score", all_dice)]:
        summary.add_data(name, round(float(arr.mean()),4), round(float(arr.std()),4),
                         round(float(arr.min()),4), round(float(arr.max()),4))
    summary.add_data("Gap (PA - Dice)",
                     round(float((all_pixel_acc - all_dice).mean()), 4),
                     round(float((all_pixel_acc - all_dice).std()),  4),
                     round(float((all_pixel_acc - all_dice).min()),  4),
                     round(float((all_pixel_acc - all_dice).max()),  4))
    wandb.log({"task2_6/metric_summary": summary})

    print(f"Task 2.6 complete. {count} sample images + full val metrics logged.")
    wandb.finish()


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