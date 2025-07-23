import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
from PIL import Image

class COCOVisualizer:
    def __init__(self, coco_dataset_dir):
        self.dataset_dir = Path(coco_dataset_dir)
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load annotations
        self.train_ann_file = self.dataset_dir / "annotations" / "train.json"
        self.val_ann_file = self.dataset_dir / "annotations" / "val.json"
        
        if self.train_ann_file.exists():
            with open(self.train_ann_file, 'r') as f:
                self.train_data = json.load(f)
        else:
            self.train_data = None
            
        if self.val_ann_file.exists():
            with open(self.val_ann_file, 'r') as f:
                self.val_data = json.load(f)
        else:
            self.val_data = None
    
    def get_category_info(self, data):
        """Create category id to name mapping"""
        category_map = {}
        for cat in data['categories']:
            category_map[cat['id']] = cat['name']
        return category_map
    
    def visualize_sample_images(self, split='train', num_samples=5):
        """Visualize sample images with bounding boxes"""
        if split == 'train' and self.train_data:
            data = self.train_data
            img_dir = self.dataset_dir / "train2017"
        elif split == 'val' and self.val_data:
            data = self.val_data
            img_dir = self.dataset_dir / "val2017"
        else:
            print(f"No data available for {split} split")
            return
        
        category_map = self.get_category_info(data)
        
        # Get random sample of images
        images = data['images']
        sample_images = random.sample(images, min(num_samples, len(images)))
        
        # Define colors for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_map)))
        color_map = {cat_id: colors[i] for i, cat_id in enumerate(category_map.keys())}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, img_info in enumerate(sample_images):
            if idx >= 6:  # Limit to 6 images for 2x3 grid
                break
                
            # Load image
            img_path = img_dir / img_info['file_name']
            if not img_path.exists():
                print(f"Image not found: {img_path}")
                continue
                
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get annotations for this image
            img_annotations = [ann for ann in data['annotations'] 
                             if ann['image_id'] == img_info['id']]
            
            # Plot image
            ax = axes[idx]
            ax.imshow(image)
            ax.set_title(f"{split.capitalize()} - {img_info['file_name']}\n"
                        f"Objects: {len(img_annotations)}")
            ax.axis('off')
            
            # Draw bounding boxes
            for ann in img_annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                category_name = category_map.get(category_id, f"class_{category_id}")
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, 
                    edgecolor=color_map[category_id],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(bbox[0], bbox[1] - 5, 
                       f"{category_name}", 
                       color=color_map[category_id],
                       fontsize=10, 
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(len(sample_images), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{split}_samples_visualization.png", 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {self.plots_dir / f'{split}_samples_visualization.png'}")
    
    def analyze_dataset_statistics(self):
        """Analyze and visualize dataset statistics"""
        stats = {}
        
        for split, data in [('train', self.train_data), ('val', self.val_data)]:
            if data is None:
                continue
                
            category_map = self.get_category_info(data)
            
            # Count annotations per category
            category_counts = {cat_id: 0 for cat_id in category_map.keys()}
            bbox_areas = []
            annotations_per_image = []
            
            # Group annotations by image
            img_ann_count = {}
            for ann in data['annotations']:
                category_counts[ann['category_id']] += 1
                bbox_areas.append(ann['area'])
                
                img_id = ann['image_id']
                img_ann_count[img_id] = img_ann_count.get(img_id, 0) + 1
            
            annotations_per_image = list(img_ann_count.values())
            
            stats[split] = {
                'num_images': len(data['images']),
                'num_annotations': len(data['annotations']),
                'category_counts': category_counts,
                'category_map': category_map,
                'avg_annotations_per_image': np.mean(annotations_per_image),
                'bbox_areas': bbox_areas
            }
        
        # Create visualization
        self.plot_statistics(stats)
        return stats
    
    def plot_statistics(self, stats):
        """Create statistical plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Category distribution comparison
        if 'train' in stats and 'val' in stats:
            categories = list(stats['train']['category_map'].values())
            train_counts = [stats['train']['category_counts'][cat_id] 
                          for cat_id in stats['train']['category_map'].keys()]
            val_counts = [stats['val']['category_counts'][cat_id] 
                        for cat_id in stats['val']['category_map'].keys()]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[0,0].bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
            axes[0,0].bar(x + width/2, val_counts, width, label='Val', alpha=0.8)
            axes[0,0].set_xlabel('Categories')
            axes[0,0].set_ylabel('Number of Annotations')
            axes[0,0].set_title('Category Distribution')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(categories, rotation=45, ha='right')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Bounding box area distribution
        if 'train' in stats:
            axes[0,1].hist(stats['train']['bbox_areas'], bins=50, alpha=0.7, 
                          label='Train', color='blue')
        if 'val' in stats:
            axes[0,1].hist(stats['val']['bbox_areas'], bins=50, alpha=0.7, 
                          label='Val', color='orange')
        axes[0,1].set_xlabel('Bounding Box Area (pixels²)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Bounding Box Area Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Dataset summary
        splits = []
        images = []
        annotations = []
        avg_ann_per_img = []
        
        for split, data in stats.items():
            splits.append(split.capitalize())
            images.append(data['num_images'])
            annotations.append(data['num_annotations'])
            avg_ann_per_img.append(data['avg_annotations_per_image'])
        
        x = np.arange(len(splits))
        axes[1,0].bar(x, images, alpha=0.8, label='Images')
        axes[1,0].set_xlabel('Split')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Number of Images per Split')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(splits)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Annotations per split
        axes[1,1].bar(x, annotations, alpha=0.8, label='Annotations', color='green')
        axes[1,1].set_xlabel('Split')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Number of Annotations per Split')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(splits)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "dataset_statistics.png", 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Statistics plot saved to: {self.plots_dir / 'dataset_statistics.png'}")
    
    def verify_conversion_integrity(self):
        """Verify the conversion was successful"""
        issues = []
        
        for split, data in [('train', self.train_data), ('val', self.val_data)]:
            if data is None:
                continue
                
            img_dir = self.dataset_dir / f"{split}2017"
            
            # Check if all images exist
            missing_images = []
            for img_info in data['images']:
                img_path = img_dir / img_info['file_name']
                if not img_path.exists():
                    missing_images.append(img_info['file_name'])
            
            if missing_images:
                issues.append(f"{split}: Missing {len(missing_images)} images")
            
            # Check annotation integrity
            invalid_bboxes = []
            for ann in data['annotations']:
                bbox = ann['bbox']
                if any(val < 0 for val in bbox):
                    invalid_bboxes.append(ann['id'])
                elif bbox[2] <= 0 or bbox[3] <= 0:  # width or height <= 0
                    invalid_bboxes.append(ann['id'])
            
            if invalid_bboxes:
                issues.append(f"{split}: {len(invalid_bboxes)} invalid bboxes")
        
        # Print verification results
        if issues:
            print("⚠️  Conversion Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Conversion verification passed! No issues found.")
        
        return len(issues) == 0

def visualize_coco_dataset(coco_dataset_dir="coco_dataset"):
    """Main function to visualize and verify COCO dataset"""
    visualizer = COCOVisualizer(coco_dataset_dir)
    
    print("🔍 Verifying conversion integrity...")
    is_valid = visualizer.verify_conversion_integrity()
    
    if is_valid:
        print("\n📊 Analyzing dataset statistics...")
        stats = visualizer.analyze_dataset_statistics()
        
        print("\n🖼️  Creating sample visualizations...")
        print("Visualizing training samples...")
        visualizer.visualize_sample_images('train', num_samples=6)
        
        print("Visualizing validation samples...")
        visualizer.visualize_sample_images('val', num_samples=6)
        
        # Print summary
        print("\n📋 Dataset Summary:")
        for split, data in stats.items():
            print(f"  {split.capitalize()}:")
            print(f"    - Images: {data['num_images']}")
            print(f"    - Annotations: {data['num_annotations']}")
            print(f"    - Avg annotations/image: {data['avg_annotations_per_image']:.2f}")
            print(f"    - Categories: {len(data['category_map'])}")
    else:
        print("❌ Please fix the conversion issues before proceeding with training.")

# Usage
if __name__ == "__main__":
    # Run visualization after dataset conversion
    visualize_coco_dataset("coco_dataset")