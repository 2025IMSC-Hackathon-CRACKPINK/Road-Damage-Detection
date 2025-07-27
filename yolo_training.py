#!/usr/bin/env python3

import os
import yaml
import shutil
import random
import torch
from glob import glob
from ultralytics import YOLO

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

def main():
    model = YOLO('yolo11x.pt')
    # model = YOLO('/home/kwdahun/2025IMSC-Hackathon-CRACKPINK/runs/road_damage_detection8/weights/best.pt')

    gpus = [1, 2, 3, 4, 5, 6, 7]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    num_gpus = len(gpus)
    base_batch_size = 12
    total_batch_size = base_batch_size * num_gpus

    results = model.train(
        data='roboflow_dataset_3/data.yaml',
        epochs=600,
        imgsz=640,
        batch=total_batch_size,
        device=gpus,
        project='runs',
        name='roboflow_road_damage_3_detection',
        workers=8 * num_gpus,
        patience=30,
        save_period=10,
        amp=True,
        cache=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()