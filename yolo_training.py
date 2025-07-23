#!/usr/bin/env python3
"""
YOLO Training for Road Damage Detection

This script trains a YOLO model for road damage detection including:
- Pothole
- Alligator Crack  
- Transverse Crack
- Longitudinal Crack

Based on the YOLO training notebook approach with multi-GPU support.
"""

import os
import yaml
import shutil
import random
import torch
import argparse
import sys
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Check if ultralytics is installed, if not install it
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO


def check_cuda():
    """Check CUDA availability and GPU count"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return torch.cuda.is_available()


def load_model(model_path='yolo11m.pt'):
    """Load YOLO model"""
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    return model


def setup_multi_gpu(gpu_ids=None):
    """Setup multi-GPU environment"""
    if gpu_ids is None:
        # Use all available GPUs except GPU 0 (reserved for other tasks)
        available_gpus = list(range(1, torch.cuda.device_count()))
        if not available_gpus:
            available_gpus = [0]  # Fallback to GPU 0 if no others available
        gpu_ids = available_gpus
    
    # Set CUDA_VISIBLE_DEVICES
    gpu_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    
    print(f"Using GPUs: {gpu_ids}")
    print(f"CUDA_VISIBLE_DEVICES: {gpu_str}")
    
    return gpu_ids


def train_model(model, data_yaml='yolo_augmented_dataset/data.yaml', 
                epochs=100, batch_size=16, gpu_ids=None, project='runs',
                name='road_damage_detection_multigpu', patience=15):
    """Train YOLO model with multi-GPU support"""
    
    if gpu_ids is None:
        gpu_ids = setup_multi_gpu()
    
    num_gpus = len(gpu_ids)
    total_batch_size = batch_size * num_gpus
    
    print(f"Training configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Base batch size: {batch_size}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Workers: {8 * num_gpus}")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: Data file {data_yaml} not found!")
        return None
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=total_batch_size,
            device=gpu_ids,
            project=project,
            name=name,
            workers=8 * num_gpus,
            patience=patience,
            save_period=10,
            amp=True,
            cache=True,
            cos_lr=True,
            verbose=True
        )
        
        print("Training completed successfully!")
        return results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None


def evaluate_model(model_path=None, data_yaml='yolo_dataset/data.yaml', device=0):
    """Evaluate trained model"""
    print("Evaluating model...")
    
    if model_path is None:
        # Find the latest training run
        train_dirs = sorted(glob('runs/detect/road_damage_detection*'), key=os.path.getmtime, reverse=True)
        if train_dirs:
            model_path = f"{train_dirs[0]}/weights/best.pt"
            print(f"Loading best model from: {model_path}")
        else:
            print("No trained model found!")
            return None
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None
    
    eval_model = YOLO(model_path)
    
    # Evaluate on test set
    try:
        results = eval_model.val(
            data=data_yaml,
            split='test',
            device=device,
            verbose=True
        )
        
        # Print metrics
        print(f"\nEvaluation Results:")
        print(f"mAP@0.5:      {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision:    {results.box.mp:.4f}")
        print(f"Recall:       {results.box.mr:.4f}")
        print(f"Detailed results saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        return None


def save_best_model(checkpoint_dir='yolo_checkpoints'):
    """Save the best model from training runs"""
    train_dirs = sorted(glob('runs/detect/road_damage_detection*'), key=os.path.getmtime, reverse=True)
    
    if not train_dirs:
        print("No training runs found!")
        return None
    
    best_model_path = f"{train_dirs[0]}/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        return None
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    dest_path = f"{checkpoint_dir}/best_road_damage.pt"
    shutil.copy(best_model_path, dest_path)
    
    print(f"Model saved: {dest_path}")
    return dest_path


def load_best_model():
    """Load the best model from training runs or checkpoints"""
    # Try to load from latest training run
    train_dirs = sorted(glob('runs/detect/road_damage_detection*'), key=os.path.getmtime, reverse=True)
    if train_dirs:
        best_model_path = f"{train_dirs[0]}/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"Loading model from: {best_model_path}")
            return YOLO(best_model_path)
    
    # Try to load from checkpoints
    checkpoint_path = 'yolo_checkpoints/best_road_damage.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        return YOLO(checkpoint_path)
    
    # Fallback to base model
    print("No trained model found. Using base model.")
    return YOLO('yolo11m.pt')


def run_inference(model=None, test_images_pattern='yolo_dataset/test/images/*.jpg', 
                  num_images=5, device=0, save_results=True):
    """Run inference on test images"""
    if model is None:
        model = load_best_model()
    
    test_images = glob(test_images_pattern)[:num_images]
    
    if not test_images:
        print(f"No test images found with pattern: {test_images_pattern}")
        return
    
    print(f"Running inference on {len(test_images)} test images...")
    
    class_names = ['Pothole', 'Alligator Crack', 'Transverse Crack', 'Longitudinal Crack']
    
    for i, img_path in enumerate(test_images):
        try:
            # Run inference
            result = model(img_path, device=device)
            
            # Display result if matplotlib is available and not in headless mode
            if save_results:
                annotated_img = result[0].plot()
                
                # Save annotated image
                output_dir = 'inference_results'
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/result_{Path(img_path).stem}.jpg"
                cv2.imwrite(output_path, annotated_img)
                print(f"Saved result to: {output_path}")
            
            # Print detection details
            boxes = result[0].boxes
            print(f"\nDetections in {Path(img_path).name}:")
            if boxes is not None and len(boxes) > 0:
                for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
                    print(f"  {class_names[int(cls)]}: {conf:.3f}")
            else:
                print("  No detections found")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Training for Road Damage Detection')
    parser.add_argument('--mode', choices=['train', 'eval', 'inference', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--data', default='yolo_augmented_dataset/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Base batch size per GPU')
    parser.add_argument('--model', default='yolo11m.pt',
                       help='Model to load')
    parser.add_argument('--gpus', nargs='+', type=int,
                       help='GPU IDs to use (default: all except 0)')
    parser.add_argument('--project', default='runs',
                       help='Project directory')
    parser.add_argument('--name', default='road_damage_detection_multigpu',
                       help='Experiment name')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Check CUDA
    cuda_available = check_cuda()
    if not cuda_available:
        print("CUDA not available. Training may be slow.")
    
    if args.mode in ['train', 'all']:
        print("\n=== TRAINING ===")
        # Load model
        model = load_model(args.model)
        
        # Train model
        results = train_model(
            model=model,
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gpu_ids=args.gpus,
            project=args.project,
            name=args.name,
            patience=args.patience
        )
        
        if results:
            # Save best model
            save_best_model()
    
    if args.mode in ['eval', 'all']:
        print("\n=== EVALUATION ===")
        # Evaluate model
        eval_results = evaluate_model()
    
    if args.mode in ['inference', 'all']:
        print("\n=== INFERENCE ===")
        # Run inference
        run_inference()
    
    print("\nScript completed!")


if __name__ == "__main__":
    main()