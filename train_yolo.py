#!/usr/bin/env python3
"""
YOLO Training Script - Road Damage Detection Training
Enhanced with comprehensive data augmentation for crack detection

Features:
- Extensive data augmentation (no rotation to preserve crack directions)
- Support for longitudinal and transverse crack detection
- Customizable augmentation parameters
- GPU/CPU auto-detection
- Comprehensive logging

Usage Examples:
  # Basic training with default augmentations
  python train_yolo.py --data yolo_dataset/data.yaml --epochs 50

  # Training with custom augmentation settings
  python train_yolo.py --epochs 100 --hsv-gain 0.8 --mixup 0.3 --copy-paste 0.5

  # Training without augmentation
  python train_yolo.py --no-augment --epochs 30
"""

import os
import sys
import torch
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 for Road Damage Detection')
    parser.add_argument('--data', type=str, default='yolo_dataset/data.yaml', 
                        help='Path to data config file')
    parser.add_argument('--model', type=str, default='yolo11l.pt', 
                        help='Base model path (yolo11l.pt, yolo11m.pt, yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None, 
                        help='Batch size (auto-detected based on GPU if not specified)')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='Image size')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (auto-detected if not specified)')
    
    # Augmentation control arguments
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation (default)')
    aug_group.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--hsv-gain', type=float, default=0.5,
                        help='HSV augmentation gain multiplier (0.0-1.0)')
    parser.add_argument('--translate', type=float, default=0.2,
                        help='Translation augmentation fraction (0.0-1.0)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale augmentation gain (0.0-1.0)')
    parser.add_argument('--mixup', type=float, default=0.15,
                        help='Mixup augmentation probability (0.0-1.0)')
    parser.add_argument('--copy-paste', type=float, default=0.3,
                        help='Copy-paste augmentation probability (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Handle augmentation flag
    if args.no_augment:
        args.augment = False
    
    # Print system information
    print("=" * 60)
    print("YOLO11 Road Damage Detection Training")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {memory_gb:.1f} GB")
    
    # Check if dataset config exists
    if not os.path.exists(args.data):
        print(f"Error: Dataset config file not found: {args.data}")
        print("Please ensure the dataset is prepared and the config file exists.")
        sys.exit(1)
    
    # Check if base model exists
    if not os.path.exists(args.model):
        print(f"Error: Base model not found: {args.model}")
        print("Please download the model file or check the path.")
        sys.exit(1)
    
    # Initialize YOLO11 model - start fresh from base model
    print(f"\nStarting fresh training from base model: {args.model}")
    model = YOLO(args.model)
    
    # Configure training parameters based on available resources
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if args.batch is None:
        if torch.cuda.is_available():
            batch_size = 16  # Match reference file
            workers = 8
        else:
            batch_size = 8
            workers = 4
    else:
        batch_size = args.batch
        workers = 8 if torch.cuda.is_available() else 4
    
    # Training configuration with extensive augmentations (no rotation for crack direction preservation)
    training_config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': batch_size,
        'imgsz': args.imgsz,
        'device': device,
        'workers': workers,
        'project': 'road_damage_detection',
        'name': 'yolo11_training_extended',
        'exist_ok': True,
    }
    
    # Add augmentation parameters if enabled
    if args.augment:
        augmentation_config = {
            # Geometric augmentations (no rotation to preserve crack directions)
            'degrees': 0.0,              # No rotation for directional crack preservation
            'translate': args.translate, # Translation augmentation
            'scale': args.scale,         # Scale augmentation (±50%)
            'shear': 0.1,               # Shear augmentation (10°)
            'perspective': 0.0005,       # Perspective transformation
            'flipud': 0.5,              # Vertical flip probability
            'fliplr': 0.5,              # Horizontal flip probability
            
            # Color/appearance augmentations
            'hsv_h': 0.015 * args.hsv_gain,  # HSV Hue augmentation
            'hsv_s': 0.7 * args.hsv_gain,    # HSV Saturation augmentation
            'hsv_v': 0.4 * args.hsv_gain,    # HSV Value/brightness augmentation
            
            # Noise and quality augmentations
            'bgr': 0.0,                 # BGR channel shuffle (disabled for road images)
            
            # Mosaic and mixup augmentations
            'mosaic': 1.0,              # Mosaic augmentation probability
            'mixup': args.mixup,        # Mixup augmentation probability
            'copy_paste': args.copy_paste, # Copy-paste augmentation probability
            
            # Advanced augmentations
            'erasing': 0.4,             # Random erasing probability
            'crop_fraction': 1.0,       # Crop fraction for training
            
            # Additional augmentations for robustness
            'auto_augment': 'randaugment',  # AutoAugment policy
            'augment': True,            # Enable augmentation
        }
        training_config.update(augmentation_config)
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {workers}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Project: {training_config['project']}")
    print(f"  Name: {training_config['name']}")
    
    if args.augment:
        print(f"\nAugmentation Settings:")
        print(f"  Translation: ±{args.translate*100:.1f}%")
        print(f"  Scale: ±{args.scale*100:.1f}%")
        print(f"  HSV gain: {args.hsv_gain:.2f}")
        print(f"  Mixup probability: {args.mixup:.2f}")
        print(f"  Copy-paste probability: {args.copy_paste:.2f}")
        print(f"  Mosaic: Enabled")
        print(f"  Auto-augment: RandAugment")
        print(f"  Flip (H/V): 50%/50%")
        print(f"  Shear: ±0.1°")
        print(f"  Perspective: 0.0005")
        print(f"  Random erasing: 40%")
        print(f"  Rotation: DISABLED (preserves crack directions)")
    else:
        print(f"\nAugmentation: DISABLED")
    
    print("Ready for training...")
    
    # Start YOLO11 training
    print("\n" + "=" * 60)
    print("Starting YOLO11 training from scratch...")
    print(f"Training for {args.epochs} epochs with configuration above")
    print("=" * 60)
    
    try:
        results = model.train(**training_config)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Results saved at: {results.save_dir}")
        print("=" * 60)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"\nBest model weights saved at: {results.save_dir}/weights/best.pt")
        print(f"Training results saved at: {results.save_dir}/results.csv")
        print(f"Training plots saved at: {results.save_dir}/")
        
        return results.save_dir
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

if __name__ == "__main__":
    try:
        save_dir = main()
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results directory: {save_dir}")
        print(f"🏆 Best model: {save_dir}/weights/best.pt")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)