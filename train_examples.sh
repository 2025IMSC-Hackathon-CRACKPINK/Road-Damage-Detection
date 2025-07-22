#!/bin/bash
# Enhanced YOLO11 Road Damage Detection Training Examples
# With comprehensive data augmentation strategies

echo "🚧 YOLO11 Road Damage Detection Training Examples 🚧"
echo "===================================================="

echo "1. Basic training with default augmentations:"
echo "python train_yolo.py --data yolo_dataset/data.yaml --epochs 30"
echo ""

echo "2. Aggressive augmentation for small datasets:"
echo "python train_yolo.py --epochs 50 --hsv-gain 0.8 --mixup 0.3 --copy-paste 0.5 --translate 0.3 --scale 0.7"
echo ""

echo "3. Conservative augmentation for large datasets:"
echo "python train_yolo.py --epochs 100 --hsv-gain 0.3 --mixup 0.1 --copy-paste 0.2 --translate 0.1 --scale 0.3"
echo ""

echo "4. No augmentation (baseline comparison):"
echo "python train_yolo.py --no-augment --epochs 30"
echo ""

echo "5. High-quality training with extended epochs:"
echo "python train_yolo.py --model yolo11l.pt --epochs 200 --batch 8 --hsv-gain 0.6 --mixup 0.2"
echo ""

echo "6. Fast training with nano model:"
echo "python train_yolo.py --model yolo11n.pt --epochs 50 --batch 32"
echo ""

echo "7. Custom model path and device:"
echo "python train_yolo.py --data custom_dataset/data.yaml --device cpu --batch 4"
echo ""

echo ""
echo "📋 Augmentation Features:"
echo "✅ Translation: Spatial position variations"
echo "✅ Scale: Size variations" 
echo "✅ HSV: Color/lighting variations"
echo "✅ Horizontal/Vertical Flips: Orientation diversity"
echo "✅ Shear: Perspective variations"
echo "✅ Mosaic: Multi-image composition"
echo "✅ Mixup: Image blending for robustness"
echo "✅ Copy-Paste: Object placement diversity"
echo "✅ Random Erasing: Occlusion robustness"
echo "✅ RandAugment: Automated policy selection"
echo "❌ Rotation: DISABLED (preserves crack directions)"
echo ""

echo "🔧 Available Models:"
echo "  - yolo11n.pt (nano - fastest, lower accuracy)"
echo "  - yolo11m.pt (medium - balanced speed/accuracy)" 
echo "  - yolo11l.pt (large - best accuracy, default)"
echo ""

echo "📊 Monitoring:"
echo "  - Results: road_damage_detection/yolo11_training_extended/"
echo "  - Best model: road_damage_detection/yolo11_training_extended/weights/best.pt"
echo "  - Training plots: road_damage_detection/yolo11_training_extended/*.png"
echo "  - Metrics: road_damage_detection/yolo11_training_extended/results.csv"
echo ""

echo "🎯 Crack Detection Optimizations:"
echo "  - No rotation preserves longitudinal/transverse crack integrity"
echo "  - HSV augmentation handles different lighting/weather conditions"
echo "  - Translation/scale augmentation helps with various crack sizes"
echo "  - Mixup improves generalization across different road surfaces"
