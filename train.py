import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
from src.data.dataset import ISICImageDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.vit_classifier import ImageOnlyViT
from src.models.resnet50_classifier import ResNet50Classifier
from src.models.resnet101_classifier import ResNet101Classifier
from src.models.resnet152_classifier import ResNet152Classifier
from src.models.densenet_classifier import DenseNetClassifier
from src.models.convnext_classifier import ConvNeXtClassifier
from src.losses.cross_entropy_loss import WeightedCrossEntropyLoss

DEFAULT_CONFIG = {
    'batch_size': 64,
    'num_workers': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'img_size': 224,
    'gamma': 2.0,
    'patience': 10,
    'model_type': 'vit', 
    'dropout': 0.5,
}

def train_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Training'):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets.unsqueeze(1))
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc='Validating'):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        with autocast('cuda'):
            outputs = model(images)
            
            if torch.isnan(outputs).any():
                print("⚠️  WARNING: NaN detected in model outputs!")
                outputs = torch.nan_to_num(outputs, nan=0.0)
            
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    if torch.isnan(all_preds).any():
        print("⚠️  WARNING: NaN in predictions, replacing with 0.5")
        all_preds = torch.nan_to_num(all_preds, nan=0.5)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_targets, all_preds)
    return total_loss / len(loader), auc

def save_checkpoint(model, optimizer, epoch, val_auc, save_path, model_type='vit'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc,
        'model_type': model_type,
    }, save_path)
    print(f"💾 Checkpoint saved to {save_path}")


def main(args):
    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / manifest['split_id']
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = ISICImageDataset(
        catalog_path=args.catalog,
        manifest_path=args.manifest,
        split='train',
        img_dir=args.img_dir,
        transform=get_train_transforms(args.img_size)
    )
    val_dataset = ISICImageDataset(
        catalog_path=args.catalog,
        manifest_path=args.manifest,
        split='val',
        img_dir=args.img_dir,
        transform=get_val_transforms(args.img_size)
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Model Selection
    print(f"\nCreating model: {args.model_type.upper()}")
    if args.model_type == 'resnet50':
        model = ResNet50Classifier(
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'resnet101':
        model = ResNet101Classifier(
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'resnet152':
        model = ResNet152Classifier(
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'densenet121':
        model = DenseNetClassifier(
            model_name=args.model_type,
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'densenet169':
        model = DenseNetClassifier(
            model_name=args.model_type,
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'convnext_tiny':
        model = ConvNeXtClassifier(
            model_name=args.model_type,
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'convnext_base':
        model = ConvNeXtClassifier(
            model_name=args.model_type,
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    else:  # vit (default)
        model = ImageOnlyViT(
            pretrained=True,
            num_classes=1,
            dropout=args.dropout
        ).to(device)
    
    # Loss (calculate pos_weight from training data)
    train_catalog = pd.read_csv(args.catalog)
    train_catalog = train_catalog[train_catalog['image'].isin(manifest['train_ids'])]
    pos_count = train_catalog['target'].sum()
    neg_count = len(train_catalog) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    print(f"⚖️  Class imbalance - Pos: {pos_count}, Neg: {neg_count}, Weight: {pos_weight.item():.2f}")
    
    # Loss Selection
    if args.loss_type == 'cross_entropy':
        criterion = WeightedCrossEntropyLoss(pos_weight=pos_weight)
        print(f"📉 Using Cross Entropy Loss")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Mixed Precision Scaler
    scaler = GradScaler('cuda')
    
    # Training Loop
    best_val_auc = 0
    patience_counter = 0
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, args.grad_clip)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_auc:.4f}")
        
        # Scheduler step
        scheduler.step()
        
        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_auc,
                output_dir / 'models' / 'best_checkpoint.pth',
                model_type=args.model_type
            )
            print(f"New best model! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        
        # Save epoch checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_auc,
                output_dir / 'models' / f'checkpoint_epoch_{epoch + 1}.pth',
                model_type=args.model_type
            )
    
    # Save training config
    config = {
        'split_id': manifest['split_id'],
        'data_hash': manifest['data_hash'],
        'model_type': args.model_type,
        'loss_type': args.loss_type,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'img_size': args.img_size,
        'gamma': 2.0,
        'dropout': args.dropout,
        'best_val_auc': best_val_auc,
    }
    with open(output_dir / 'models' / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"🏆 Best Val AUC: {best_val_auc:.4f}")
    print(f"📁 Checkpoint: {output_dir / 'models' / 'best_checkpoint.pth'}")

# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Classifier (ViT, ResNet50, or ResNet101)')
    parser.add_argument('--catalog', type=str, required=True, help='Path to catalog CSV')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest JSON')
    parser.add_argument('--img_dir', type=str, default='data/raw', help='Image directory')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--model_type', type=str, default='vit', choices=[
        'vit', 
        'resnet50', 'resnet101', 'resnet152',
        'densenet121', 'densenet169',
        'convnext_tiny', 'convnext_base'
    ],
                        help='Model architecture')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', choices=['cross_entropy'],
                        help='Loss function: cross_entropy')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient Clipping')
    args = parser.parse_args()
    main(args)