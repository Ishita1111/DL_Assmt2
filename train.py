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

# Import your custom modules
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def save_split_weights(model, save_dir="checkpoints"):
    """Splits the multi-task model into 3 separate files for the autograder."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Classifier.pth (Backbone + Classification Head)
    torch.save({
        'backbone': model.backbone.state_dict(),
        'classifier_head': model.classifier.state_dict()
    }, os.path.join(save_dir, "classifier.pth"))
    
    # 2. Localizer.pth (Bounding Box Head only)
    torch.save(model.locator.state_dict(), os.path.join(save_dir, "localizer.pth"))
    
    # 3. UNet.pth (Segmentation Decoder only)
    torch.save(model.segmenter.state_dict(), os.path.join(save_dir, "unet.pth"))


def train_one_epoch(model, dataloader, optimizer, criteria, device, epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    running_seg_loss = 0.0

    # criteria is a dictionary containing our 3 loss functions
    cls_criterion = criteria['cls']
    bbox_criterion = criteria['bbox']
    seg_criterion = criteria['seg']

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, cls_targets, bbox_targets, seg_targets in progress_bar:
        # Added non_blocking=True for faster CPU -> GPU data transfers
        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        bbox_targets = bbox_targets.to(device, non_blocking=True).float()
        seg_targets = seg_targets.to(device, non_blocking=True).long() 

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Single pass yielding a dictionary of outputs
        outputs = model(images)
        cls_logits = outputs['classification']
        bbox_preds = outputs['localization']
        seg_masks = outputs['segmentation']

        # Calculate individual losses
        loss_cls = cls_criterion(cls_logits, cls_targets)
        loss_bbox = bbox_criterion(bbox_preds, bbox_targets)
        loss_seg = seg_criterion(seg_masks, seg_targets)

        # ---> UPGRADED: Multi-task Loss Weights <---
        # Boosted classification weight to help improve the F1 score!
        w_cls, w_bbox, w_seg = 2.5, 5.0, 1.0 

        # Unified Loss Formulation
        total_loss = (w_cls * loss_cls) + (w_bbox * loss_bbox) + (w_seg * loss_seg)

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += total_loss.item()
        running_cls_loss += loss_cls.item()
        running_bbox_loss += loss_bbox.item()
        running_seg_loss += loss_seg.item()

        progress_bar.set_postfix({'Total Loss': total_loss.item()})

    # Averages for the epoch
    epoch_metrics = {
        "train/total_loss": running_loss / len(dataloader),
        "train/cls_loss": running_cls_loss / len(dataloader),
        "train/bbox_loss": running_bbox_loss / len(dataloader),
        "train/seg_loss": running_seg_loss / len(dataloader),
    }
    return epoch_metrics

def calculate_batch_iou(preds, targets):
    """Calculates IoU for a batch of bounding box predictions [cx, cy, w, h]"""
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
def validate(model, dataloader, criteria, device, epoch):
    model.eval()
    running_loss = 0.0
    
    cls_criterion = criteria['cls']
    bbox_criterion = criteria['bbox']
    seg_criterion = criteria['seg']

    # Metric accumulators
    all_cls_preds = []
    all_cls_targets = []
    
    total_dice_score = 0.0
    total_iou = 0.0
    correct_detections_50 = 0 
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")

    for images, cls_targets, bbox_targets, seg_targets in progress_bar:
        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        bbox_targets = bbox_targets.to(device, non_blocking=True).float()
        seg_targets = seg_targets.to(device, non_blocking=True).long()

        # Forward pass: Single pass yielding a dictionary of outputs
        outputs = model(images)
        cls_logits = outputs['classification']
        bbox_preds = outputs['localization']
        seg_masks = outputs['segmentation']
        
        # ---------------- LOSS CALCULATIONS ----------------
        loss_cls = cls_criterion(cls_logits, cls_targets)
        loss_bbox = bbox_criterion(bbox_preds, bbox_targets)
        loss_seg = seg_criterion(seg_masks, seg_targets)

        # ---> UPGRADED: Multi-task Loss Weights <---
        w_cls, w_bbox, w_seg = 2.5, 5.0, 1.0 
        total_loss = (w_cls * loss_cls) + (w_bbox * loss_bbox) + (w_seg * loss_seg)
        running_loss += total_loss.item()
        
        # ---------------- METRIC CALCULATIONS ----------------
        # 1. Classification (Accumulate for F1)
        _, cls_pred_labels = torch.max(cls_logits, 1)
        all_cls_preds.extend(cls_pred_labels.cpu().numpy())
        all_cls_targets.extend(cls_targets.cpu().numpy())
        
        # 2. Bounding Box (mAP / Mean IoU)
        ious = calculate_batch_iou(bbox_preds, bbox_targets)
        total_iou += ious.sum().item()
        correct_detections_50 += (ious > 0.5).sum().item()
        
        # 3. Segmentation (Multi-Class Dice Score)
        _, seg_pred_masks = torch.max(seg_masks, 1)
        batch_dice = 0.0
        smooth = 1e-6
        for c in range(3):
            pred_c = (seg_pred_masks == c)
            target_c = (seg_targets == c)
            intersection = (pred_c & target_c).float().sum(dim=(1, 2))
            union = pred_c.float().sum(dim=(1, 2)) + target_c.float().sum(dim=(1, 2))
            dice_c = (2. * intersection + smooth) / (union + smooth)
            batch_dice += dice_c.mean().item()
            
        total_dice_score += (batch_dice / 3.0) 
        total_samples += images.size(0)

        progress_bar.set_postfix({'Val Loss': total_loss.item()})

    # ---------------- FINALIZE METRICS ----------------
    macro_f1 = f1_score(all_cls_targets, all_cls_preds, average='macro')
    mean_iou = total_iou / total_samples
    map_50 = correct_detections_50 / total_samples
    mean_dice = total_dice_score / len(dataloader)

    epoch_metrics = {
        "val/total_loss": running_loss / len(dataloader),
        "val/macro_f1": macro_f1,
        "val/bbox_mIoU": mean_iou,
        "val/bbox_mAP_50": map_50,
        "val/seg_dice": mean_dice
    }
    return epoch_metrics

def main():
    # 1. Initialize Weights & Biases
    wandb.init(
        project="da6401-assignment-2",
        config={
            "learning_rate": 1e-4,
            "epochs": 50, # ---> UPGRADED: Let the model train longer! <---
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

    # 2. DataLoaders 
    dataset_path = "./data/oxford-iiit-pet" 
    train_loader, val_loader = get_dataloaders(root_dir=dataset_path, batch_size=config.batch_size)

    # 3. Model Setup
    model = MultiTaskPerceptionModel(num_classes=37, num_seg_classes=3).to(device)
    
    # 4. Loss Functions Dictionary
    criteria = {
        'cls': nn.CrossEntropyLoss(),
        'bbox': IoULoss(), 
        'seg': nn.CrossEntropyLoss() 
    }

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # ---> UPGRADED: Added a Learning Rate Scheduler <---
    # This will cut the learning rate in half every 15 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # ---------------------------------------------------------------------------
    # 6. Training Loop Setup with "Global Memory" & Composite Scoring
    
    # Force absolute paths so it ALWAYS saves to the correct folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    metric_file = os.path.join(checkpoint_dir, "best_val_score.txt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if a previous high score exists
    if os.path.exists(metric_file):
        with open(metric_file, 'r') as f:
            best_val_score = float(f.read().strip())
        print(f"\n📈 Loaded GLOBAL best Composite Score: {best_val_score:.4f} from previous runs.")
        print("We will only overwrite the weights if we beat this score!\n")
    else:
        best_val_score = 0.0 # We want HIGHEST score, so we start at 0
        print("\n🏁 No previous high score found. Starting fresh!\n")
    # ---------------------------------------------------------------------------

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criteria, device, epoch)
        val_metrics = validate(model, val_loader, criteria, device, epoch)

        # ---------------------------------------------------------------------------
        # THE NEW EVALUATION LOGIC
        
        # 1. Calculate the average of your 3 primary metrics
        f1 = val_metrics["val/macro_f1"]
        mAP = val_metrics["val/bbox_mAP_50"]
        dice = val_metrics["val/seg_dice"]
        composite_score = (f1 + mAP + dice) / 3.0
        
        # Log everything to W&B, including the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({**train_metrics, **val_metrics, "val/composite_score": composite_score, "learning_rate": current_lr, "epoch": epoch})
        
        # 2. Check if the new composite score is HIGHER than the best
        if composite_score > best_val_score:
            print(f"\n🌟 NEW GLOBAL BEST FOUND! Composite Score improved from {best_val_score:.4f} to {composite_score:.4f}")
            print(f"   (F1: {f1:.4f} | mAP: {mAP:.4f} | Dice: {dice:.4f})")
            best_val_score = composite_score
            
            # Save the 3 autograder files using the absolute path
            save_split_weights(model, save_dir=checkpoint_dir)
            
            # Save the text file so we remember it next time!
            with open(metric_file, 'w') as f:
                f.write(str(best_val_score))
        else:
            print(f"\n❌ Composite Score ({composite_score:.4f}) did not beat global best ({best_val_score:.4f}).")
            
        # Optional: Save a full checkpoint every 5 epochs just for your own backup
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"multitask_epoch_{epoch}_backup.pth"))

        # ---> UPGRADED: Step the scheduler at the very end of the epoch <---
        scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    main()