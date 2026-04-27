import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import ISICImageDataset
from src.data.transforms import get_val_transforms
from src.models.vit_classifier import ImageOnlyViT
from src.models.resnet50_classifier import ResNet50Classifier
from src.models.resnet101_classifier import ResNet101Classifier
from src.models.resnet152_classifier import ResNet152Classifier
from src.models.convnext_classifier import ConvNeXtClassifier
from src.models.densenet_classifier import DenseNetClassifier
from src.utils.cache import InferenceCache

def get_predictions(model, loader, device, temperature=None):
    model.eval()
    all_probs = []
    all_targets = []
    all_image_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            image_ids = batch['image_id']
            logits = model(images)
            if temperature is not None:
                logits = logits / temperature
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(targets)
            all_image_ids.extend(image_ids)
    all_probs = torch.cat(all_probs).cpu().numpy().flatten()
    all_targets = torch.cat(all_targets).cpu().numpy().flatten()
    return all_probs, all_targets, all_image_ids

def main(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.config) as f:
        train_config = json.load(f)
    
    model_type = train_config.get('model_type', 'vit')
    print(f"️  Detected model type: {model_type.upper()}")
    
    temperature = None
    if args.calibration and Path(args.calibration).exists():
        with open(args.calibration) as f:
            calib = json.load(f)
            temperature = calib['temperature']
            print(f"Using temperature: {temperature:.4f}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if model_type == 'resnet50':
        model = ResNet50Classifier(pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    elif model_type == 'resnet101':
        model = ResNet101Classifier(pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    elif model_type == 'resnet152':
        model = ResNet152Classifier(pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    elif model_type == 'densenet121':
        model = DenseNetClassifier(model_name=model_type, pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    elif model_type == 'densenet169':
        model = DenseNetClassifier(model_name=model_type, pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    elif model_type == 'convnext_tiny':
        model = ConvNeXtClassifier(model_name=model_type, pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    elif model_type == 'convnext_base':
        model = ConvNeXtClassifier(model_name=model_type, pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    else:
        model = ImageOnlyViT(pretrained=False, num_classes=1, dropout=train_config.get('dropout', 0.5)).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f" Val AUC: {checkpoint['val_auc']:.4f}")
    
    cache = InferenceCache(cache_dir=args.cache_dir)
    
    if args.all_splits:
        splits = ['train', 'val', 'test']
    elif args.splits:
        splits = args.splits.split(',')
    else:
        splits = ['val', 'test']
    
    print(f" Running inference on splits: {splits}")
    for split in splits:
        print(f"\n{'='*60}")
        print(f" Running inference on {split.upper()} split...")
        print(f"{'='*60}")
        cached = cache.load(args.checkpoint, args.manifest, split, temperature)
        if cached[0] is not None and not args.force:
            print(f" Using cached inference for {split}")
            continue
        dataset = ISICImageDataset(
            catalog_path=args.catalog,
            manifest_path=args.manifest,
            split=split,
            img_dir=args.img_dir,
            transform=get_val_transforms(train_config['img_size'])
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        probs, targets, image_ids = get_predictions(model, loader, device, temperature)
        cache.save(probs, targets, image_ids, args.checkpoint, args.manifest, split, temperature)
        print(f"   Samples: {len(probs)}")
        print(f"   Malignant rate: {targets.mean():.2%}")
        print(f"   Mean probability: {probs.mean():.4f}")
    print(f"\n Inference complete! Cached results in {args.cache_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Inference and Cache Results')
    parser.add_argument('--catalog', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--calibration', type=str, default=None)
    parser.add_argument('--img_dir', type=str, default='data/raw')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--all_splits', action='store_true')
    parser.add_argument('--splits', type=str, default='val,test')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    main(args)