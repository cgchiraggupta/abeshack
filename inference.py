import torch
import os
import glob
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model(model_name, num_classes=10, device="cuda"):
    if model_name == "deeplabv3plus":
        from models.deeplabv3plus import get_model
        model_path = "checkpoints/best_model_deeplabv3plus.pth"
    elif model_name == "unet":
        from models.unet import get_model
        model_path = "checkpoints/best_model_unet.pth"
    elif model_name == "pspnet":
        from models.pspnet import get_model
        model_path = "checkpoints/best_model_pspnet.pth"
    elif model_name == "fpn":
        from models.fpn import get_model
        model_path = "checkpoints/best_model_fpn.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = get_model(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️  Model weights not found at {model_path}, using random initialization")
    
    model.eval()
    return model

def preprocess_image(image_path, size=512):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    image_resized = cv2.resize(image, (size, size))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = np.transpose(image_normalized, (2, 0, 1))
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).float()
    
    return image_tensor, original_h, original_w, image_resized

def postprocess_mask(pred_mask, original_h, original_w):
    pred_mask = torch.softmax(pred_mask, dim=1)
    pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()
    
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                  (original_w, original_h), 
                                  interpolation=cv2.INTER_NEAREST)
    
    return pred_mask_resized

def create_color_mask(mask, colors):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        color_mask[mask == class_id] = colors[class_id]
    return color_mask

def create_overlay(image, color_mask, alpha=0.5):
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay

def run_inference(model_name="deeplabv3plus", test_dir="data/testImages", output_dir="inference_results"):
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 512
    
    colors = np.array([
        [34, 139, 34],    # 0: Trees
        [0, 255, 0],      # 1: Lush Bushes
        [189, 183, 107],  # 2: Dry Grass
        [160, 82, 45],    # 3: Dry Bushes
        [105, 105, 105],  # 4: Ground Clutter
        [255, 0, 255],    # 5: Flowers
        [139, 69, 19],    # 6: Logs
        [128, 128, 128],  # 7: Rocks
        [210, 180, 140],  # 8: Landscape
        [135, 206, 235],  # 9: Sky
    ], dtype=np.uint8)
    
    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    
    if not test_images:
        print(f"Error: No test images found in {test_dir}")
        return
    
    print(f"Running inference with {model_name} on {len(test_images)} test images")
    
    model = load_model(model_name, NUM_CLASSES, DEVICE)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    
    for img_path in tqdm(test_images, desc="Processing images"):
        try:
            image_tensor, original_h, original_w, image_resized = preprocess_image(img_path, IMG_SIZE)
            image_tensor = image_tensor.to(DEVICE)
            
            with torch.no_grad():
                output = model(image_tensor)
            
            pred_mask = postprocess_mask(output, original_h, original_w)
            
            original_image = cv2.imread(img_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            color_mask = create_color_mask(pred_mask, colors)
            overlay = create_overlay(original_image, color_mask, alpha=0.5)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(color_mask)
            axes[1].set_title("Predicted Mask")
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            
            plt.suptitle(f"{model_name.upper()} - {os.path.basename(img_path)}", fontsize=14)
            plt.tight_layout()
            
            output_filename = os.path.join(output_dir, model_name, 
                                          f"{os.path.splitext(os.path.basename(img_path))[0]}_result.png")
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            cv2.imwrite(os.path.join(output_dir, model_name, 
                                    f"{os.path.splitext(os.path.basename(img_path))[0]}_mask.png"), 
                       cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\n✅ Inference completed!")
    print(f"Results saved to: {os.path.join(output_dir, model_name)}")
    
    create_legend(colors, class_names, output_dir, model_name)

def create_legend(colors, class_names, output_dir, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    for i, (color, name) in enumerate(zip(colors, class_names)):
        ax.add_patch(plt.Rectangle((0.1, 0.9 - i*0.08), 0.1, 0.06, 
                                  facecolor=color/255, edgecolor='black'))
        ax.text(0.25, 0.93 - i*0.08, name, fontsize=12, 
                verticalalignment='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Class Legend - {model_name.upper()}", fontsize=14)
    
    legend_path = os.path.join(output_dir, model_name, "class_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Class legend saved to: {legend_path}")

def run_all_models():
    models = ["deeplabv3plus", "unet", "pspnet", "fpn"]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Running inference with {model_name.upper()}")
        print(f"{'='*60}")
        try:
            run_inference(model_name)
        except Exception as e:
            print(f"Error with {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--model', type=str, default='deeplabv3plus', 
                       choices=['deeplabv3plus', 'unet', 'pspnet', 'fpn', 'all'],
                       help='Model to use for inference (default: deeplabv3plus)')
    parser.add_argument('--test_dir', type=str, default='data/testImages',
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        run_all_models()
    else:
        run_inference(args.model, args.test_dir, args.output_dir)