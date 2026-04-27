import pandas as pd
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json

class ISICImageDataset(Dataset):
    def __init__(self, catalog_path, manifest_path, split, img_dir, transform=None, is_external=False):
        self.catalog = pd.read_csv(catalog_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.split = split
        self.is_external = is_external
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        split_ids = manifest[f'{split}_ids']
        self.catalog = self.catalog[self.catalog['image'].isin(split_ids)].reset_index(drop=True)
        print(f" Loaded {len(self.catalog)} images for {split} split")
    
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        # Load image
        if self.is_external:
            # For external datasets, try multiple path formats
            img_path = Path(row['file_path'])
            # If path doesn't start with 'data/', prepend it
            if not str(img_path).startswith('data'):
                img_path = Path('data') / img_path
            # Try to load
            img = cv2.imread(str(img_path))
        else:
            # For ISIC, prepend img_dir
            img_path = self.img_dir / row['file_path']
            img = cv2.imread(str(img_path))
        
        # Handle missing images gracefully
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return {
            'image': img,
            'target': torch.tensor(row['target'], dtype=torch.float32),
            'image_id': row['image']
        }