import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import os

from data.pets_dataset import get_dataloaders
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def save_split_weights(model, save_dir="checkpoints"):
    """Three files expected by the autograder: classifier.pth bundles backbone + head."""
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'backbone': model.backbone.state_dict(),
        'classifier_head': model.classifier.state_dict()
    }, os.path.join(save_dir, "classifier.pth"))

    torch.save(model.locator.state_dict(), os.path.join(save_dir, "localizer.pth"))

    torch.save(model.segmenter.state_dict(), os.path.join(save_dir, "unet.pth"))


def train_one_epoch(model, dataloader, optimizer, loss_bundle, device, epoch):
    model.train()
    sum_loss = 0.0
    sum_cls = 0.0
    sum_box = 0.0
    sum_seg = 0.0

    # cls / bbox / seg — same keys as in main()
    cls_loss_fn = loss_bundle['cls']
    box_loss_fn = loss_bundle['bbox']
    seg_loss_fn = loss_bundle['seg']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, cls_targets, bbox_targets, seg_targets in pbar:
        # non_blocking helps when pin_memory=True on the loader
        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        bbox_targets = bbox_targets.to(device, non_blocking=True).float()
        seg_targets = seg_targets.to(device, non_blocking=True).long()

        optimizer.zero_grad()

        # one forward; bbox tensor is already in pixel space (see multitask forward)
        out = model(images)
        cls_logits = out['classification']
        bbox_preds = out['localization']
        seg_logits = out['segmentation']

        loss_cls = cls_loss_fn(cls_logits, cls_targets)
        loss_bbox = box_loss_fn(bbox_preds, bbox_targets)
        loss_seg = seg_loss_fn(seg_logits, seg_targets)

        # hand-tuned weights — cls bumped a bit for F1
        wt_cls, wt_box, wt_seg = 2.5, 5.0, 1.0
        total_loss = wt_cls * loss_cls + wt_box * loss_bbox + wt_seg * loss_seg

        total_loss.backward()
        optimizer.step()

        sum_loss += total_loss.item()
        sum_cls += loss_cls.item()
        sum_box += loss_bbox.item()
        sum_seg += loss_seg.item()

        pbar.set_postfix({'Total Loss': total_loss.item()})

    epoch_metrics = {
        "train/total_loss": sum_loss / len(dataloader),
        "train/cls_loss": sum_cls / len(dataloader),
        "train/bbox_loss": sum_box / len(dataloader),
        "train/seg_loss": sum_seg / len(dataloader),
    }
    return epoch_metrics

def calculate_batch_iou(preds, targets):
    """IoU per sample, boxes as cx, cy, w, h."""
    pred_x1 = preds[:, 0] - preds[:, 2] / 2
    pred_y1 = preds[:, 1] - preds[:, 3] / 2
    pred_x2 = preds[:, 0] + preds[:, 2] / 2
    pred_y2 = preds[:, 1] + preds[:, 3] / 2

    target_x1 = targets[:, 0] - targets[:, 2] / 2
    target_y1 = targets[:, 1] - targets[:, 3] / 2
    target_x2 = targets[:, 0] + targets[:, 2] / 2
    target_y2 = targets[:, 1] + targets[:, 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = preds[:, 2] * preds[:, 3]
    target_area = targets[:, 2] * targets[:, 3]
    union_area = pred_area + target_area - inter_area

    return inter_area / (union_area + 1e-6)

@torch.no_grad()
def validate(model, dataloader, loss_bundle, device, epoch):
    model.eval()
    sum_loss = 0.0

    cls_loss_fn = loss_bundle['cls']
    box_loss_fn = loss_bundle['bbox']
    seg_loss_fn = loss_bundle['seg']

    cls_pred_list = []
    cls_true_list = []

    total_dice_score = 0.0
    total_iou = 0.0
    correct_detections_50 = 0 
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")

    for images, cls_targets, bbox_targets, seg_targets in pbar:
        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        bbox_targets = bbox_targets.to(device, non_blocking=True).float()
        seg_targets = seg_targets.to(device, non_blocking=True).long()

        out = model(images)
        cls_logits = out['classification']
        bbox_preds = out['localization']
        seg_logits = out['segmentation']

        loss_cls = cls_loss_fn(cls_logits, cls_targets)
        loss_bbox = box_loss_fn(bbox_preds, bbox_targets)
        loss_seg = seg_loss_fn(seg_logits, seg_targets)

        wt_cls, wt_box, wt_seg = 2.5, 5.0, 1.0
        total_loss = wt_cls * loss_cls + wt_box * loss_bbox + wt_seg * loss_seg
        sum_loss += total_loss.item()

        # macro F1 needs all preds — sklearn at end of epoch
        _, pred_cls = torch.max(cls_logits, 1)
        cls_pred_list.extend(pred_cls.cpu().numpy())
        cls_true_list.extend(cls_targets.cpu().numpy())

        ious = calculate_batch_iou(bbox_preds, bbox_targets)
        total_iou += ious.sum().item()
        correct_detections_50 += (ious > 0.5).sum().item()

        # mean Dice over trimap classes {0,1,2}, averaged per batch then over batches
        _, pred_seg = torch.max(seg_logits, 1)
        batch_dice = 0.0
        eps = 1e-6
        for c in range(3):
            pred_c = (pred_seg == c)
            target_c = (seg_targets == c)
            intersection = (pred_c & target_c).float().sum(dim=(1, 2))
            union = pred_c.float().sum(dim=(1, 2)) + target_c.float().sum(dim=(1, 2))
            dice_c = (2.0 * intersection + eps) / (union + eps)
            batch_dice += dice_c.mean().item()

        total_dice_score += batch_dice / 3.0
        total_samples += images.size(0)

        pbar.set_postfix({'Val Loss': total_loss.item()})

    macro_f1 = f1_score(cls_true_list, cls_pred_list, average='macro')
    mean_iou = total_iou / total_samples
    map_50 = correct_detections_50 / total_samples
    # dice above is summed per batch (mean over 3 classes); divide by num batches
    mean_dice = total_dice_score / len(dataloader)

    epoch_metrics = {
        "val/total_loss": sum_loss / len(dataloader),
        "val/macro_f1": macro_f1,
        "val/bbox_mIoU": mean_iou,
        "val/bbox_mAP_50": map_50,
        "val/seg_dice": mean_dice
    }
    return epoch_metrics

def main():
    wandb.init(
        project="da6401-assignment-2",
        config={
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "architecture": "VGG11-UNet-MultiTask"
        }
    )
    config = wandb.config

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    data_root = "./data/oxford-iiit-pet"
    train_loader, val_loader = get_dataloaders(root_dir=data_root, batch_size=config.batch_size)

    model = MultiTaskPerceptionModel(num_classes=37, num_seg_classes=3).to(device)

    loss_bundle = {
        'cls': nn.CrossEntropyLoss(),
        'bbox': IoULoss(),
        'seg': nn.CrossEntropyLoss()
    }

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # halve LR every 15 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    here = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(here, "checkpoints")
    metric_file = os.path.join(checkpoint_dir, "best_val_score.txt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(metric_file):
        with open(metric_file, 'r') as f:
            best_val_score = float(f.read().strip())
        print(f"\nResuming best composite score from disk: {best_val_score:.4f}")
        print("(checkpoints only update if we beat that.)\n")
    else:
        best_val_score = 0.0
        print("\nNo prior best score file — starting from 0.\n")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_bundle, device, epoch)
        val_metrics = validate(model, val_loader, loss_bundle, device, epoch)

        f1 = val_metrics["val/macro_f1"]
        mAP = val_metrics["val/bbox_mAP_50"]
        dice = val_metrics["val/seg_dice"]
        # simple average of the three headline metrics
        composite_score = (f1 + mAP + dice) / 3.0

        lr_now = optimizer.param_groups[0]['lr']
        wandb.log({**train_metrics, **val_metrics, "val/composite_score": composite_score, "learning_rate": lr_now, "epoch": epoch})

        if composite_score > best_val_score:
            print(f"\nNew best composite: {best_val_score:.4f} -> {composite_score:.4f}")
            print(f"   F1 {f1:.4f} | mAP@0.5 {mAP:.4f} | dice {dice:.4f}")
            best_val_score = composite_score

            save_split_weights(model, save_dir=checkpoint_dir)

            with open(metric_file, 'w') as f:
                f.write(str(best_val_score))
        else:
            print(f"\nComposite {composite_score:.4f} (best so far {best_val_score:.4f}).")

        # full-state backup only; grading uses save_split_weights
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"multitask_epoch_{epoch}_backup.pth"))

        scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    main()