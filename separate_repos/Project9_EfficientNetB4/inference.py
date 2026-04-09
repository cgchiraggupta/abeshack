import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.efficientnet_b4 import EfficientNetB4Segmentation
import matplotlib.pyplot as plt
from PIL import Image

class OffRoadInference:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize inference pipeline
        """
        self.device = device
        
        # Load model
        self.model = EfficientNetB4Segmentation(num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Inference transforms
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Class names and colors for visualization
        self.class_names = [
            'Trees', 'Lush Bushes', 'Grass', 'Dirt', 'Sand', 
            'Water', 'Rocks', 'Bushes', 'Mud', 'Background'
        ]
        
        self.class_colors = [
            (34, 139, 34),    # Trees - Forest Green
            (0, 100, 0),      # Lush Bushes - Dark Green
            (124, 252, 0),    # Grass - Lawn Green
            (139, 69, 19),    # Dirt - Saddle Brown
            (238, 203, 173),  # Sand - Burlywood
            (30, 144, 255),   # Water - Dodger Blue
            (128, 128, 128),  # Rocks - Gray
            (0, 128, 0),      # Bushes - Green
            (101, 67, 33),    # Mud - Dark Brown
            (0, 0, 0)         # Background - Black
        ]
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        """
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        original_h, original_w = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        return image_tensor, original_h, original_w
    
    def predict_single(self, image_path, save_path=None):
        """
        Perform inference on a single image
        """
        # Preprocess image
        image_tensor, original_h, original_w = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Resize prediction to original dimensions
        pred_resized = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored segmentation mask
        colored_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        for class_idx in range(10):
            colored_mask[pred_resized == class_idx] = self.class_colors[class_idx]
        
        # Read original image for visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Create overlay (50% transparency)
        overlay = cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)
        
        if save_path:
            # Save results
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save segmentation mask
            mask_save_path = save_path.replace('.png', '_mask.png')
            cv2.imwrite(mask_save_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            
            # Save overlay
            overlay_save_path = save_path.replace('.png', '_overlay.png')
            cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Create side-by-side visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(colored_mask)
            axes[1].set_title('Segmentation Mask')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay (50% transparency)')
            axes[2].axis('off')
            
            plt.suptitle('Off-Road Terrain Segmentation - EfficientNet-B4', fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Results saved to: {save_path}")
        
        return pred_resized, colored_mask, overlay
    
    def predict_batch(self, input_dir, output_dir, image_extensions=['.jpg', '.jpeg', '.png']):
        """
        Perform batch inference on all images in a directory
        """
        # Create output directories
        masks_dir = os.path.join(output_dir, 'masks')
        overlays_dir = os.path.join(output_dir, 'overlays')
        visualizations_dir = os.path.join(output_dir, 'visualizations')
        
        for dir_path in [masks_dir, overlays_dir, visualizations_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
        
        print(f"Found {len(image_paths)} images for inference")
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Get base filename
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Define output paths
                viz_path = os.path.join(visualizations_dir, f"{name_without_ext}_result.png")
                mask_path = os.path.join(masks_dir, f"{name_without_ext}_mask.png")
                overlay_path = os.path.join(overlays_dir, f"{name_without_ext}_overlay.png")
                
                # Perform inference
                pred_mask, colored_mask, overlay = self.predict_single(image_path, viz_path)
                
                # Save individual components
                cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        print(f"\nBatch inference completed!")
        print(f"Results saved to: {output_dir}")
        print(f"- Masks: {masks_dir}")
        print(f"- Overlays: {overlays_dir}")
        print(f"- Visualizations: {visualizations_dir}")
    
    def predict_video(self, video_path, output_path, fps=30, frame_skip=1):
        """
        Perform inference on video frames
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, Frames: {total_frames}")
        
        frame_count = 0
        processed_count = 0
        
        with tqdm(total=total_frames//frame_skip, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for model
                frame_resized = cv2.resize(frame_rgb, (512, 512))
                
                # Apply transforms
                transformed = self.transform(image=frame_resized)
                image_tensor = transformed['image'].unsqueeze(0).to(self.device)
                
                # Perform inference
                with torch.no_grad():
                    output = self.model(image_tensor)
                    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                
                # Resize prediction to original dimensions
                pred_resized = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                for class_idx in range(10):
                    colored_mask[pred_resized == class_idx] = self.class_colors[class_idx]
                
                # Create overlay
                overlay = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)
                
                # Write frame to output video
                out.write(overlay)
                
                processed_count += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\nVideo processing completed!")
        print(f"Processed {processed_count} frames")
        print(f"Output saved to: {output_path}")

def main():
    """
    Main function for inference
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Off-Road Terrain Segmentation Inference')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input image/video/directory path')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'batch', 'video'], help='Inference mode')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = OffRoadInference(args.model, device=args.device)
    
    if args.mode == 'image':
        # Single image inference
        output_path = os.path.join(args.output, 'result.png')
        inference.predict_single(args.input, output_path)
    
    elif args.mode == 'batch':
        # Batch inference
        inference.predict_batch(args.input, args.output)
    
    elif args.mode == 'video':
        # Video inference
        output_path = os.path.join(args.output, 'output_video.mp4')
        inference.predict_video(args.input, output_path)

if __name__ == "__main__":
    main()