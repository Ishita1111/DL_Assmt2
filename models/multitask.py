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
        
        # backbone is sharing across all 3 tasks
        self.backbone = VGG11Backbone(pretrained=pretrained, use_bn=use_bn)
        
        #   heads 
        self.classifier = ClassificationHead(num_classes=num_classes)
        self.locator = RegressionHead()
        self.segmenter = UNetDecoder(num_classes=num_seg_classes)

        if download_weights:
            # weight store
            p_cls = "checkpoints/classifier.pth"
            p_box = "checkpoints/localizer.pth"
            p_seg = "checkpoints/unet.pth"
            os.makedirs("checkpoints", exist_ok=True)

            # gdrive weight loading
            print("Downloading weights from Google Drive...")
            gdown.download(id="16F5q7AGYELb09JIy4lgacICizM9PuVJc", output=p_cls, quiet=False)
            gdown.download(id="1rrSHIr0Y8I-D0wY0VWV9FOLIZWjMlXnA", output=p_box, quiet=False)
            gdown.download(id="13ekfanq3G5B_mVLGGMWyTFyPQT0OMW9R", output=p_seg, quiet=False)
                
            
            print("Loading weights into model...")
            
            chk = torch.load(p_cls, map_location="cpu")
            self.backbone.load_state_dict(chk['backbone'])
            
            # Dictionary translator fix for old weights
            cls_weights = chk['classifier_head']
            if '1.weight' in cls_weights:
                fixed_cls_weights = {
                    'classifier.2.weight': cls_weights['1.weight'],
                    'classifier.2.bias': cls_weights['1.bias'],
                    'classifier.5.weight': cls_weights['4.weight'],
                    'classifier.5.bias': cls_weights['4.bias'],
                    'classifier.8.weight': cls_weights['7.weight'],
                    'classifier.8.bias': cls_weights['7.bias'],
                }
                self.classifier.load_state_dict(fixed_cls_weights)
            else:
                self.classifier.load_state_dict(cls_weights)

            self.locator.load_state_dict(torch.load(p_box, map_location="cpu"))
            self.segmenter.load_state_dict(torch.load(p_seg, map_location="cpu"))
                
            print("Successfully loaded all pretrained weights!")

    def forward(self, x):
        # shared backbone
        bottleneck, skip_features = self.backbone(x)
        
        # breeding label
        class_logits = self.classifier(bottleneck)
        
       
        raw_box = self.locator(bottleneck) #bounding box
        
        
        _, _, H, W = x.shape 
        normalized_box = raw_box * 224.0  #recovering by  multiplying by 224
        
        #scaling the normalized box to the actual imgae space
        scale_tensor = torch.tensor([W, H, W, H], device=raw_box.device)
        image_space_box = normalized_box * scale_tensor
        # -----------------
        
        
        seg_mask = self.segmenter(bottleneck, skip_features) #segmen mask
        
        return {
            'classification': class_logits,
            'localization': image_space_box, 
            'segmentation': seg_mask
        }
