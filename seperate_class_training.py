import os
import shutil
import random
import yaml
import argparse
from glob import glob
from ultralytics import YOLO


def separate_classes_by_country():
    """Separate dataset by class from country directories."""
    root_data_dir = './data'
    output_base = './class_split'  # Class-specific output directory
    
    # Clean up existing class_split directory
    if os.path.exists(output_base):
        print(f"Cleaning up existing {output_base} directory...")
        shutil.rmtree(output_base)
    
    os.makedirs(output_base, exist_ok=True)

    country_dirs = [d for d in os.listdir(root_data_dir) if d.startswith('country_')]

    # Process each class (0-3)
    for cls_id in range(4):
        img_out = os.path.join(output_base, f'class_{cls_id}/images')
        lbl_out = os.path.join(output_base, f'class_{cls_id}/labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        # Process each country directory
        for country in country_dirs:
            original_img_dir = os.path.join(root_data_dir, country, 'images')
            original_lbl_dir = os.path.join(root_data_dir, country, 'labels')

            label_files = sorted(glob(f"{original_lbl_dir}/*.txt"))

            for lbl_path in label_files:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()

                # Filter labels for current class only and convert class ID to 0
                filtered = []
                for line in lines:
                    if line.strip().startswith(str(cls_id) + ' '):
                        # Convert class ID to 0 for single-class training
                        parts = line.strip().split()
                        parts[0] = '0'  # Change class ID to 0
                        filtered.append(' '.join(parts) + '\n')

                if filtered:
                    # Copy corresponding image file
                    base = os.path.basename(lbl_path).replace('.txt', '')
                    for ext in ['.jpg', '.png']:
                        img_path = os.path.join(original_img_dir, base + ext)
                        if os.path.exists(img_path):
                            shutil.copy(img_path, os.path.join(img_out, base + ext))
                            break

                    with open(os.path.join(lbl_out, base + '.txt'), 'w') as f:
                        f.writelines(filtered)

    print("Class separation completed for all countries!")


def split_dataset(base_dir, class_id, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test splits."""
    img_dir = os.path.join(base_dir, f'class_{class_id}/images')
    lbl_dir = os.path.join(base_dir, f'class_{class_id}/labels')

    img_paths = sorted(glob(f'{img_dir}/*.jpg') + glob(f'{img_dir}/*.png'))
    random.shuffle(img_paths)

    n = len(img_paths)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train': img_paths[:train_end],
        'val': img_paths[train_end:val_end],
        'test': img_paths[val_end:]
    }

    for split in splits:
        os.makedirs(f'{base_dir}/class_{class_id}/images/{split}', exist_ok=True)
        os.makedirs(f'{base_dir}/class_{class_id}/labels/{split}', exist_ok=True)

    for split, paths in splits.items():
        for img_path in paths:
            base = os.path.basename(img_path).rsplit('.', 1)[0]
            lbl_path = os.path.join(lbl_dir, base + '.txt')

            shutil.copy(img_path, f'{base_dir}/class_{class_id}/images/{split}/')
            shutil.copy(lbl_path, f'{base_dir}/class_{class_id}/labels/{split}/')

    print(f"class_{class_id} split into train/val/test completed ({n} samples)")


def create_data_yaml(base_dir, class_id, class_name):
    """Create YAML configuration file for each class."""
    # Get absolute path to avoid path resolution issues
    abs_base_dir = os.path.abspath(base_dir)
    
    data = {
        'train': f'{abs_base_dir}/class_{class_id}/images/train',
        'val': f'{abs_base_dir}/class_{class_id}/images/val',
        'test': f'{abs_base_dir}/class_{class_id}/images/test',
        'nc': 1,
        'names': [class_name]
    }

    yaml_path = os.path.join(base_dir, f'class_{class_id}/data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"data.yaml created for class_{class_id}")


def train_models(device, class_id):
    """Train YOLO model for a specific class."""
    print(f"Training model for class_{class_id} on device: {device}")
    
    model = YOLO('yolov8m.pt')
    model.train(
        data=f'class_split/class_{class_id}/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,  # Use specified device
        project='cls_yolo_runs',
        name=f'class_{class_id}'
    )
    print(f"Training completed for class_{class_id}")


def main():
    parser = argparse.ArgumentParser(description='Train separate YOLO models for each road damage class')
    parser.add_argument('--device', type=str, default='0', 
                       help='Device to use for training (e.g., "0", "1", "cpu"). Default: "0"')
    parser.add_argument('--class-id', type=int, required=True,
                       help='Class ID to train (0-3). Required.')
    parser.add_argument('--skip-data-prep', action='store_false',
                       help='Perform data preparation steps (class separation, splitting, yaml creation). Default: skip data prep.')
    
    args = parser.parse_args()
    
    if args.class_id < 0 or args.class_id > 3:
        print("Error: class-id must be between 0 and 3")
        return
    
    if not args.skip_data_prep:
        print("Step 1: Separating classes by country...")
        separate_classes_by_country()
        
        print("Step 2: Splitting datasets into train/val/test...")
        for cls_id in range(4):
            split_dataset('./class_split', cls_id)
        
        print("Step 3: Creating YAML configuration files...")
        class_names = ['Pothole', 'Alligator Crack', 'Transverse Crack', 'Longitudinal Crack']
        for cls in range(4):
            create_data_yaml('./class_split', cls, class_names[cls])
    
    print(f"Step 4: Training model for class_{args.class_id}...")
    train_models(args.device, args.class_id)
    
    print(f"Training completed for class_{args.class_id}!")


if __name__ == "__main__":
    main()