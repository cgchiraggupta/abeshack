import os
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from models.attention_unet import get_model

class TerrainInference:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = get_model(num_classes=self.config['model']['num_classes'])
        self.model = self.model.to(self.device)
        
        checkpoint_path = self.config['inference']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_colors = {
            0: [255, 0, 0],      # Class 0: Red
            1: [0, 255, 0],      # Class 1: Green
            2: [0, 0, 255],      # Class 2: Blue
            3: [255, 255, 0],    # Class 3: Yellow
            4: [255, 0, 255],    # Class 4: Magenta
            5: [0, 255, 255],    # Class 5: Cyan
            6: [128, 0, 0],      # Class 6: Maroon
            7: [0, 128, 0],      # Class 7: Dark Green
            8: [0, 0, 128],      # Class 8: Navy
            9: [128, 128, 0]     # Class 9: Olive
        }
        
        self.class_names = {
            0: "Trees",
            1: "Lush Bushes",
            2: "Dry Bushes",
            3: "Grass",
            4: "Dirt",
            5: "Gravel",
            6: "Rocks",
            7: "Sand",
            8: "Water",
            9: "Sky"
        }
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        image = image.resize((512, 512), Image.BILINEAR)
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_size
    
    def predict(self, image_path):
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1)
        
        prediction = prediction.squeeze().cpu().numpy()
        
        prediction_resized = Image.fromarray(prediction.astype(np.uint8))
        prediction_resized = prediction_resized.resize(original_size, Image.NEAREST)
        prediction = np.array(prediction_resized)
        
        return prediction
    
    def create_colored_mask(self, prediction):
        h, w = prediction.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in self.class_colors.items():
            mask = prediction == class_idx
            colored_mask[mask] = color
        
        return colored_mask
    
    def create_overlay(self, original_image, colored_mask, alpha=0.5):
        original_np = np.array(original_image)
        overlay = original_np.copy()
        
        mask_indices = np.any(colored_mask != [0, 0, 0], axis=-1)
        overlay[mask_indices] = (alpha * colored_mask[mask_indices] + 
                                (1 - alpha) * original_np[mask_indices]).astype(np.uint8)
        
        return overlay
    
    def save_results(self, image_path, output_dir=None):
        if output_dir is None:
            output_dir = self.config['inference']['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        basename = os.path.basename(image_path).split('.')[0]
        
        original_image = Image.open(image_path).convert('RGB')
        original_size = original_image.size
        
        prediction = self.predict(image_path)
        colored_mask = self.create_colored_mask(prediction)
        overlay = self.create_overlay(original_image, colored_mask)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(prediction, cmap='tab20', vmin=0, vmax=self.config['model']['num_classes']-1)
        axes[0, 1].set_title('Semantic Segmentation')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(colored_mask)
        axes[1, 0].set_title('Colored Mask')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (α=0.5)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{basename}_segmentation.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        class_counts = np.bincount(prediction.flatten(), minlength=self.config['model']['num_classes'])
        
        print(f"\nSegmentation results saved to: {output_path}")
        print(f"\nClass distribution:")
        for class_idx in range(self.config['model']['num_classes']):
            if class_counts[class_idx] > 0:
                percentage = (class_counts[class_idx] / prediction.size) * 100
                print(f"  {self.class_names.get(class_idx, f'Class {class_idx}')}: "
                      f"{class_counts[class_idx]} pixels ({percentage:.2f}%)")
        
        return {
            'prediction': prediction,
            'colored_mask': colored_mask,
            'overlay': overlay,
            'class_counts': class_counts.tolist(),
            'output_path': output_path
        }
    
    def process_folder(self, input_folder, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(self.config['inference']['output_dir'], 'batch_results')
        
        os.makedirs(output_folder, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])
        
        print(f"Found {len(image_files)} images in {input_folder}")
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            print(f"Processing: {image_file}")
            
            try:
                result = self.save_results(image_path, output_folder)
                results.append({
                    'image': image_file,
                    'class_counts': result['class_counts'],
                    'output_path': result['output_path']
                })
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        summary_path = os.path.join(output_folder, 'summary.csv')
        import pandas as pd
        
        summary_data = []
        for result in results:
            row = {'image': result['image']}
            for class_idx in range(self.config['model']['num_classes']):
                row[f'class_{class_idx}'] = result['class_counts'][class_idx]
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_path, index=False)
        
        print(f"\nBatch processing complete. Results saved to {output_folder}")
        print(f"Summary saved to {summary_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Inference with Attention UNet for Off-Road Terrain Segmentation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, help='Path to single image for inference')
    parser.add_argument('--folder', type=str, help='Path to folder containing images for batch inference')
    parser.add_argument('--output', type=str, help='Custom output directory')
    
    args = parser.parse_args()
    
    inference = TerrainInference(args.config)
    
    if args.image:
        inference.save_results(args.image, args.output)
    elif args.folder:
        inference.process_folder(args.folder, args.output)
    else:
        print("Please provide either --image or --folder argument")
        print("Example: python inference.py --image path/to/image.jpg")
        print("         python inference.py --folder path/to/images/")

if __name__ == "__main__":
    main()