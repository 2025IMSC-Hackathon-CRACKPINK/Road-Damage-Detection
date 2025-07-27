import os
import argparse
from ultralytics import YOLO

def main(class_num, device):
    """
    Train YOLO model for a specific class
    
    Args:
        class_num (int): Class number to train
        device (str): Device to use for training (e.g., '0', 'cpu')
    """
    model = YOLO('yolov8m.pt')
    model.train(
        data=f'class_split/class_{class_num}/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        project='cls_yolo_runs',
        name=f'class_{class_num}',
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model for specific class')
    parser.add_argument('--class_num', type=int, default=1, help='Class number to train (default: 1)')
    parser.add_argument('--device', type=str, default='0', help='Device to use for training (default: 0)')
    
    args = parser.parse_args()
    main(args.class_num, args.device)