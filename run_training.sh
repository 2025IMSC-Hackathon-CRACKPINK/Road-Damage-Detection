#!/bin/bash

# Shell script to run separate class training on different GPUs
# Each training runs in background using nohup

echo "Starting training for all classes on different GPUs..."
echo "Logs will be saved to training_class_*.log files"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training for each class on different GPUs
echo "Starting training for class 0 on GPU 1..."
nohup python seperate_class_training.py --class-id 0 --device 1 > logs/training_class_0.log 2>&1 &

echo "Starting training for class 1 on GPU 2..."
nohup python seperate_class_training.py --class-id 1 --device 2 > logs/training_class_1.log 2>&1 &

echo "Starting training for class 2 on GPU 3..."
nohup python seperate_class_training.py --class-id 2 --device 3 > logs/training_class_2.log 2>&1 &

echo "Starting training for class 3 on GPU 4..."
nohup python seperate_class_training.py --class-id 3 --device 4 > logs/training_class_3.log 2>&1 &

echo "All training processes started in background!"
echo ""
echo "To monitor progress, use:"
echo "  tail -f logs/training_class_0.log"
echo "  tail -f logs/training_class_1.log"
echo "  tail -f logs/training_class_2.log"
echo "  tail -f logs/training_class_3.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep seperate_class_training"
echo ""
echo "To kill all training processes if needed:"
echo "  pkill -f seperate_class_training.py"
