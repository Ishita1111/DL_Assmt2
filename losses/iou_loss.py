import torch
import torch.nn as nn

class IoULoss(nn.Module):
    # def __init__(self, eps=1e-6):
    #     super(IoULoss, self).__init__()
    #     self.eps = eps

    # 1. Add the reduction parameter with a default value of 'mean'
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        pred_x1 = preds[:, 0] - preds[:, 2] / 2
        pred_y1 = preds[:, 1] - preds[:, 3] / 2
        pred_x2 = preds[:, 0] + preds[:, 2] / 2
        pred_y2 = preds[:, 1] + preds[:, 3] / 2

        target_x1 = targets[:, 0] - targets[:, 2] / 2
        target_y1 = targets[:, 1] - targets[:, 3] / 2
        target_x2 = targets[:, 0] + targets[:, 2] / 2
        target_y2 = targets[:, 1] + targets[:, 3] / 2

        # Calculate intersection coordinates
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        # Calculate intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Calculate union area
        pred_area = preds[:, 2] * preds[:, 3]
        target_area = targets[:, 2] * targets[:, 3]
        union_area = pred_area + target_area - inter_area

        # Calculate IoU
        # iou = inter_area / (union_area + self.eps)
        
        # Return loss (1 - IoU)
        # return 1.0 - iou.mean()

        #--------------------------------
        # Calculate the actual IoU score
        iou = inter_area / (union_area + 1e-6)
        
        # The Loss is exactly 1.0 minus the IoU score 
        # (Perfect overlap = 1.0 IoU = 0.0 Loss)
        loss = 1.0 - iou
        
        # 2. Apply the reduction logic right before returning
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")
        #--------------------------------