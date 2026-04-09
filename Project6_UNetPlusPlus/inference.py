import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image

from models.unet_plusplus import UNetPlusPlus

class UNetPlusPlusInference:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = UNetPlusPlus(num_classes=11, deep_supervision=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.class_colors = {
            0: [0, 0, 0],        # Background - Black
            100: [34, 139, 34],   # Trees - Forest Green
            200: [0, 100, 0],     # Lush Bushes - Dark Green
            300: [139, 69, 19],   # Dry Bushes - Saddle Brown
            400: [124, 252, 0],   # Grass - Lawn Green
            500: [169, 169, 169], # Concrete - Dark Gray
            600: [105, 105, 105], # Rocks - Dim Gray
            700: [30, 144, 255],  # Water - Dodger Blue
            800: [139, 90, 43],   # Dirt - Peru
            900: [101, 67, 33],   # Mud - Dark Brown
            1000: [255, 250, 250] # Snow - Snow White
        }
        
        self.class_names = {
            0: "Background",
            100: "Trees",
            200: "Lush Bushes",
            300: "Dry Bushes",
            400: "Grass",
            500: "Concrete",
            600: "Rocks",
            700: "Water",
            800: "Dirt",
            900: "Mud",
            1000: "Snow"
        }
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return image, image_tensor, original_size
    
    def predict(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1)
            probabilities = torch.softmax(output, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        return pred.cpu().numpy()[0], confidence.cpu().numpy()[0]
    
    def apply_color_map(self, prediction):
        height, width = prediction.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            mask = prediction == class_id
            colored_mask[mask] = color
        
        return colored_mask
    
    def create_overlay(self, image, colored_mask, alpha=0.5):
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlay
    
    def save_results(self, original_image, prediction, confidence_map, output_dir, base_name):
        os.makedirs(output_dir, exist_ok=True)
        
        colored_mask = self.apply_color_map(prediction)
        overlay = self.create_overlay(original_image, colored_mask)
        
        original_resized = cv2.resize(original_image, (512, 512))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].imshow(original_resized)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(colored_mask)
        axes[0, 1].set_title('Segmentation Mask')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay (α=0.5)')
        axes[1, 0].axis('off')
        
        conf_plot = axes[1, 1].imshow(confidence_map, cmap='viridis')
        axes[1, 1].set_title('Prediction Confidence')
        axes[1, 1].axis('off')
        plt.colorbar(conf_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_mask.png'), 
                   cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_overlay.png'), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        np.save(os.path.join(output_dir, f'{base_name}_prediction.npy'), prediction)
        np.save(os.path.join(output_dir, f'{base_name}_confidence.npy'), confidence_map)
    
    def process_single_image(self, image_path, output_dir='results'):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"Processing: {base_name}")
        original_image, image_tensor, original_size = self.preprocess_image(image_path)
        prediction, confidence = self.predict(image_tensor)
        
        prediction_resized = cv2.resize(prediction.astype(np.uint8), 
                                       (original_size[1], original_size[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        confidence_resized = cv2.resize(confidence, 
                                       (original_size[1], original_size[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        
        self.save_results(original_image, prediction_resized, confidence_resized, 
                         output_dir, base_name)
        
        class_counts = np.bincount(prediction_resized.flatten(), minlength=1001)
        
        print(f"  Segmentation complete for {base_name}")
        print(f"  Saved results to {output_dir}/")
        
        return class_counts
    
    def process_batch(self, image_dir, output_dir='batch_results', image_ext='*.jpg'):
        image_paths = glob(os.path.join(image_dir, image_ext))
        image_paths.extend(glob(os.path.join(image_dir, '*.png')))
        image_paths.extend(glob(os.path.join(image_dir, '*.jpeg')))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        all_class_counts = np.zeros(1001)
        
        for image_path in tqdm(image_paths, desc="Processing batch"):
            class_counts = self.process_single_image(image_path, output_dir)
            all_class_counts += class_counts
        
        self.generate_batch_report(all_class_counts, len(image_paths), output_dir)
    
    def generate_batch_report(self, class_counts, num_images, output_dir):
        total_pixels = np.sum(class_counts)
        
        report = {
            'total_images': num_images,
            'total_pixels': int(total_pixels),
            'class_distribution': {},
            'pixel_percentages': {}
        }
        
        print("\n" + "="*50)
        print("BATCH PROCESSING REPORT")
        print("="*50)
        print(f"Total Images Processed: {num_images}")
        print(f"Total Pixels: {total_pixels:,}")
        print("\nClass Distribution:")
        print(f"{'Class':<20} {'Pixels':<15} {'Percentage':<10}")
        print("-" * 50)
        
        for class_id in sorted(self.class_names.keys()):
            if class_counts[class_id] > 0:
                pixels = class_counts[class_id]
                percentage = (pixels / total_pixels) * 100
                
                report['class_distribution'][self.class_names[class_id]] = int(pixels)
                report['pixel_percentages'][self.class_names[class_id]] = float(percentage)
                
                print(f"{self.class_names[class_id]:<20} {pixels:<15,} {percentage:.2f}%")
        
        import json
        with open(os.path.join(output_dir, 'batch_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\nDetailed report saved to {output_dir}/batch_report.json")
        
        self.plot_class_distribution(class_counts, output_dir)
    
    def plot_class_distribution(self, class_counts, output_dir):
        labels = []
        sizes = []
        
        for class_id in sorted(self.class_names.keys()):
            if class_counts[class_id] > 0:
                labels.append(self.class_names[class_id])
                sizes.append(class_counts[class_id])
        
        if not sizes:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        colors = [tuple(c/255 for c in self.class_colors[class_id]) 
                 for class_id in sorted(self.class_names.keys()) 
                 if class_counts[class_id] > 0]
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Class Distribution (Percentage)')
        
        y_pos = np.arange(len(labels))
        ax2.barh(y_pos, sizes, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Number of Pixels')
        ax2.set_title('Class Distribution (Absolute Count)')
        ax2.invert_yaxis()
        
        for i, v in enumerate(sizes):
            ax2.text(v + max(sizes)*0.01, i, f'{v:,}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='UNet++ Inference for Off-Road Terrain Segmentation')
    parser.add_argument('--model', type=str, default='unetplusplus_final.pth',
                       help='Path to trained model weights')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--batch', action='store_true',
                       help='Process batch of images from directory')
    
    args = parser.parse_args()
    
    inference = UNetPlusPlusInference(args.model)
    
    if args.batch:
        inference.process_batch(args.input, args.output)
    else:
        if os.path.isfile(args.input):
            inference.process_single_image(args.input, args.output)
        else:
            print(f"Input path {args.input} is not a valid file")

if __name__ == '__main__':
    main()