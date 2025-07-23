import json
import os
import cv2
from datetime import datetime
from pathlib import Path
import shutil
from tqdm import tqdm

class DatasetToCOCO:
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_id = 1
        self.annotation_id = 1
        self.category_mapping = {}
        
    def create_directory_structure(self):
        """Create COCO directory structure"""
        (self.target_dir / "train2017").mkdir(parents=True, exist_ok=True)
        (self.target_dir / "val2017").mkdir(parents=True, exist_ok=True)
        (self.target_dir / "annotations").mkdir(parents=True, exist_ok=True)
    
    def parse_label_file(self, label_path):
        """Parse YOLO format label file: <class> <x_center> <y_center> <width_normalized> <height_normalized>"""
        annotations = []
        
        if label_path.suffix == '.txt':
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate normalized coordinates (should be between 0 and 1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 <= width <= 1 and 0 <= height <= 1):
                            print(f"Warning: Invalid normalized coordinates in {label_path}: {line}")
                            continue
                        
                        annotations.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        
        return annotations
    
    def yolo_to_coco_bbox(self, yolo_bbox, img_width, img_height):
        """Convert YOLO format to COCO bbox format"""
        x_center, y_center, width, height = yolo_bbox
        
        # Convert normalized coordinates to pixel coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Convert to COCO format (x_min, y_min, width, height)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        
        return [x_min, y_min, width, height]
    
    def convert_dataset(self, train_ratio=0.8, val_ratio=0.2):
        """Convert dataset to COCO format"""
        self.create_directory_structure()
        
        # Initialize COCO format dictionaries
        train_coco = self.init_coco_dict()
        val_coco = self.init_coco_dict()
        
        all_countries = [d for d in self.source_dir.iterdir() if d.is_dir()]
        
        # Collect all categories
        categories = set()
        for country_dir in all_countries:
            labels_dir = country_dir / "labels"
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    annotations = self.parse_label_file(label_file)
                    for ann in annotations:
                        categories.add(ann['class_id'])
        
        # Create category mapping
        for i, cat_id in enumerate(sorted(categories)):
            self.category_mapping[cat_id] = i + 1
            category_dict = {
                "id": i + 1,
                "name": f"class_{cat_id}",  # Modify with actual class names
                "supercategory": "object"
            }
            train_coco["categories"].append(category_dict)
            val_coco["categories"].append(category_dict)
        
        # Process each country
        for country_dir in tqdm(all_countries, desc="Processing countries"):
            self.process_country(country_dir, train_coco, val_coco, train_ratio)
        
        # Save annotation files
        with open(self.target_dir / "annotations" / "train.json", 'w') as f:
            json.dump(train_coco, f, indent=2)
        
        with open(self.target_dir / "annotations" / "val.json", 'w') as f:
            json.dump(val_coco, f, indent=2)
        
        print(f"Conversion completed!")
        print(f"Train images: {len(train_coco['images'])}")
        print(f"Val images: {len(val_coco['images'])}")
        print(f"Categories: {len(train_coco['categories'])}")
    
    def process_country(self, country_dir, train_coco, val_coco, train_ratio):
        """Process images and labels for a country"""
        images_dir = country_dir / "images"
        labels_dir = country_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {country_dir.name}: missing images or labels directory")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        # Split into train/val
        num_train = int(len(image_files) * train_ratio)
        train_files = image_files[:num_train]
        val_files = image_files[num_train:]
        
        # Process train files
        for img_file in tqdm(train_files, desc=f"Processing {country_dir.name} train"):
            self.process_image(img_file, labels_dir, train_coco, "train2017")
        
        # Process val files
        for img_file in tqdm(val_files, desc=f"Processing {country_dir.name} val"):
            self.process_image(img_file, labels_dir, val_coco, "val2017")
    
    def process_image(self, img_file, labels_dir, coco_dict, split):
        """Process a single image and its annotations"""
        # Load image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            return
        
        height, width = img.shape[:2]
        
        # Copy image to target directory
        new_img_name = f"{self.image_id:012d}.jpg"
        target_img_path = self.target_dir / split / new_img_name
        shutil.copy2(img_file, target_img_path)
        
        # Add image info to COCO dict
        image_info = {
            "id": self.image_id,
            "license": 1,
            "file_name": new_img_name,
            "height": height,
            "width": width,
            "date_captured": datetime.now().isoformat()
        }
        coco_dict["images"].append(image_info)
        
        # Process annotations
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            annotations = self.parse_label_file(label_file)
            
            for ann in annotations:
                # Convert YOLO to COCO bbox
                bbox = self.yolo_to_coco_bbox(
                    [ann['x_center'], ann['y_center'], ann['width'], ann['height']],
                    width, height
                )
                
                annotation_dict = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": self.category_mapping[ann['class_id']],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": []
                }
                coco_dict["annotations"].append(annotation_dict)
                self.annotation_id += 1
        
        self.image_id += 1
    
    def init_coco_dict(self):
        """Initialize COCO format dictionary"""
        return {
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Custom dataset converted to COCO format",
                "contributor": "DETR Training",
                "url": "",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }

# Usage
if __name__ == "__main__":
    converter = DatasetToCOCO(
        source_dir="data",  # Your current data directory
        target_dir="coco_dataset"  # Output COCO format directory
    )
    converter.convert_dataset(train_ratio=0.8, val_ratio=0.2)