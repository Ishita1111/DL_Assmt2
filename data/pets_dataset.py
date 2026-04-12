import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """Oxford Pets: image + trimap + VOC box. split is trainval or test."""
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, 'images')
        self.trimaps_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xml_dir = os.path.join(root_dir, 'annotations', 'xmls')
        
        split_file = os.path.join(root_dir, 'annotations', f'{split}.txt')
        
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                # official list is 1..37; CrossEntropy wants 0..36
                class_id = int(parts[1]) - 1

                xml_path = os.path.join(self.xml_dir, f'{image_name}.xml')
                if os.path.exists(xml_path):
                    self.samples.append((image_name, class_id))

    def _parse_xml_bbox(self, xml_path, img_width, img_height):
        """VOC box -> YOLO-style cx,cy,w,h in [0,1]."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + (w / 2)
        cy = ymin + (h / 2)
        
        return [cx / img_width, cy / img_height, w / img_width, h / img_height]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, class_id = self.samples[idx]
        
        img_path = os.path.join(self.images_dir, f'{image_name}.jpg')
        image = np.array(Image.open(img_path).convert('RGB'))
        img_height, img_width = image.shape[:2]
        
        trimap_path = os.path.join(self.trimaps_dir, f'{image_name}.png')
        # PNG uses 1,2,3 -> 0,1,2 for CE
        trimap = np.array(Image.open(trimap_path)) - 1

        xml_path = os.path.join(self.xml_dir, f'{image_name}.xml')
        bbox = self._parse_xml_bbox(xml_path, img_width, img_height)
        
        if self.transform:
            # yolo = normalized cx,cy,w,h — matches RegressionHead + multitask scaling
            transformed = self.transform(
                image=image,
                mask=trimap,
                bboxes=[bbox],
                class_labels=[class_id]
            )
            image = transformed['image']
            trimap = transformed['mask']
            bbox = transformed['bboxes'][0]

        class_id = torch.tensor(class_id, dtype=torch.long)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        # mask already tensor from ToTensorV2
        trimap = trimap.clone().detach().long()
        
        return image, class_id, bbox, trimap

def get_dataloaders(root_dir, batch_size=16):
    """Helper function to create Train and Validation dataloaders."""
    
    # Standard VGG normalization
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        normalize,
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        normalize,
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Load the ENTIRE dataset from trainval (since test has no XMLs)
    train_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='trainval', transform=train_transform)
    val_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='trainval', transform=val_transform)
    
    # Create an 80/20 train/validation split
    dataset_size = len(train_dataset.samples) # Usually around 3686 images
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Assign the split samples back to the datasets
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader