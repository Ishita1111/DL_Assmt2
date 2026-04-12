import torch
import torch.nn as nn
from .vgg11 import VGG11Backbone, ClassificationHead
from .localization import RegressionHead
from .segmentation import UNetDecoder
import gdown
import os


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_classes=37, num_seg_classes=3, pretrained=False, use_bn=True, download_weights=True):
        super().__init__()

        self.backbone = VGG11Backbone(pretrained=pretrained, use_bn=use_bn)

        self.classifier = ClassificationHead(num_classes=num_classes)
        self.locator = RegressionHead()
        self.segmenter = UNetDecoder(num_classes=num_seg_classes)

        if download_weights:
            path_cls = "checkpoints/classifier.pth"
            path_box = "checkpoints/localizer.pth"
            path_seg = "checkpoints/unet.pth"
            os.makedirs("checkpoints", exist_ok=True)

            print("Downloading weights from Google Drive...")
            gdown.download(id="16F5q7AGYELb09JIy4lgacICizM9PuVJc", output=path_cls, quiet=False)
            gdown.download(id="1rrSHIr0Y8I-D0wY0VWV9FOLIZWjMlXnA", output=path_box, quiet=False)
            gdown.download(id="13ekfanq3G5B_mVLGGMWyTFyPQT0OMW9R", output=path_seg, quiet=False)

            print("Loading weights into model...")

            cls_ckpt = torch.load(path_cls, map_location="cpu")
            self.backbone.load_state_dict(cls_ckpt['backbone'])

            head_sd = cls_ckpt['classifier_head']
            if '1.weight' in head_sd:
                # older checkpoint used different Sequential indices
                remapped = {
                    'classifier.2.weight': head_sd['1.weight'],
                    'classifier.2.bias': head_sd['1.bias'],
                    'classifier.5.weight': head_sd['4.weight'],
                    'classifier.5.bias': head_sd['4.bias'],
                    'classifier.8.weight': head_sd['7.weight'],
                    'classifier.8.bias': head_sd['7.bias'],
                }
                self.classifier.load_state_dict(remapped)
            else:
                self.classifier.load_state_dict(head_sd)

            self.locator.load_state_dict(torch.load(path_box, map_location="cpu"))
            self.segmenter.load_state_dict(torch.load(path_seg, map_location="cpu"))

            print("Successfully loaded all pretrained weights!")

    def forward(self, x):
        bottleneck, skips = self.backbone(x)

        cls_logits = self.classifier(bottleneck)

        # head outputs YOLO-style box in [0,1]; targets from loader are normalized to current W,H
        box_norm = self.locator(bottleneck)

        _, _, H, W = x.shape
        box_224 = box_norm * 224.0
        whwh = torch.tensor([W, H, W, H], device=box_norm.device)
        box_px = box_224 * whwh

        seg_logits = self.segmenter(bottleneck, skips)

        return {
            'classification': cls_logits,
            'localization': box_px,
            'segmentation': seg_logits
        }
