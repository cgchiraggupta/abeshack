import torch
import os
import glob
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset.dataset import SegDataset
from metrics import dice_score, iou_score

def load_model(num_classes=10, device="cuda"):
    from models.segformer import get_model
    model_path = "checkpoints/best_model.pth"
    
    model = get_model(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️  Model weights not found at {model_path}")
        return None
    
    model.eval()
    return model

def test_model(test_dir="data/testImages", has_masks=False):
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    
    print("Testing SegFormer-B2 model")
    print("="*60)
    
    model = load_model(NUM_CLASSES, DEVICE)
    if model is None:
        return
    
    if has_masks:
        test_img_dir = os.path.join(test_dir, "Color_Images")
        test_mask_dir = os.path.join(test_dir, "Segmentation")
        
        test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.png")))
        test_masks = sorted(glob.glob(os.path.join(test_mask_dir, "*.png")))
        
        if not test_imgs or not test_masks:
            print(f"Error: No test images or masks found in {test_dir}")
            return
        
        print(f"Found {len(test_imgs)} test images with masks")
        
        test_ds = SegDataset(test_imgs, test_masks, augment=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        total_dice = 0
        total_iou = 0
        per_class_dice_acc = torch.zeros(NUM_CLASSES).to(DEVICE)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, masks in tqdm(test_loader, desc="Testing"):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                outputs = model(imgs)
                
                d_mean, d_class = dice_score(outputs, masks, NUM_CLASSES)
                total_dice += d_mean.item()
                per_class_dice_acc += d_class
                
                iou = iou_score(outputs, masks, NUM_CLASSES)
                total_iou += iou.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
        
        avg_dice = total_dice / len(test_loader)
        avg_iou = total_iou / len(test_loader)
        avg_per_class_dice = per_class_dice_acc / len(test_loader)
        
        print(f"\nTest Results for SegFormer-B2:")
        print(f"Mean Dice Score: {avg_dice:.4f}")
        print(f"Mean IoU Score: {avg_iou:.4f}")
        
        class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
                      "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
        
        print("\nPer-Class Dice Scores:")
        for i, (name, score) in enumerate(zip(class_names, avg_per_class_dice.cpu().numpy())):
            print(f"  {name:15s}: {score:.4f}")
        
        cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Test Confusion Matrix - SegFormer-B2')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('test_confusion_matrix_segformer.png', dpi=300)
        plt.close()
        
        print(f"\nConfusion matrix saved to test_confusion_matrix_segformer.png")
        
        report = classification_report(all_targets, all_preds, 
                                     target_names=class_names, 
                                     digits=4, zero_division=0)
        print(f"\nClassification Report:\n{report}")
        
        with open('test_report_segformer.txt', 'w') as f:
            f.write("Test Report - SegFormer-B2\n")
            f.write("="*50 + "\n\n")
            f.write(f"Mean Dice Score: {avg_dice:.4f}\n")
            f.write(f"Mean IoU Score: {avg_iou:.4f}\n\n")
            f.write("Per-Class Dice Scores:\n")
            for i, (name, score) in enumerate(zip(class_names, avg_per_class_dice.cpu().numpy())):
                f.write(f"  {name:15s}: {score:.4f}\n")
            f.write("\n" + report)
        
        print(f"Detailed report saved to test_report_segformer.txt")
        
        return {
            'model': 'segformer',
            'mean_dice': avg_dice,
            'mean_iou': avg_iou,
            'per_class_dice': avg_per_class_dice.cpu().numpy().tolist()
        }
    
    else:
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        
        if not test_images:
            print(f"Error: No test images found in {test_dir}")
            return
        
        print(f"Found {len(test_images)} test images (no masks available)")
        print("Running inference only...")
        
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
        
        output_dir = "test_results_segformer"
        os.makedirs(output_dir, exist_ok=True)
        
        for img_path in tqdm(test_images, desc="Processing test images"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_h, original_w = image_rgb.shape[:2]
                
                image_resized = cv2.resize(image_rgb, (512, 512))
                image_normalized = image_resized.astype(np.float32) / 255.0
                image_tensor = np.transpose(image_normalized, (2, 0, 1))
                image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(DEVICE)
                
                with torch.no_grad():
                    output = model(image_tensor)
                
                pred_mask = torch.softmax(output, dim=1)
                pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()
                
                pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                              (original_w, original_h), 
                                              interpolation=cv2.INTER_NEAREST)
                
                color_mask = np.zeros((pred_mask_resized.shape[0], pred_mask_resized.shape[1], 3), dtype=np.uint8)
                for class_id in range(len(colors)):
                    color_mask[pred_mask_resized == class_id] = colors[class_id]
                
                overlay = cv2.addWeighted(image_rgb, 0.5, color_mask, 0.5, 0)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(image_rgb)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                axes[1].imshow(color_mask)
                axes[1].set_title("Predicted Mask")
                axes[1].axis('off')
                
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis('off')
                
                plt.suptitle(f"SegFormer-B2 - {os.path.basename(img_path)}", fontsize=14)
                plt.tight_layout()
                
                output_filename = os.path.join(output_dir, 
                                              f"{os.path.splitext(os.path.basename(img_path))[0]}_result.png")
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"\n✅ Test inference completed!")
        print(f"Results saved to: {output_dir}")
        
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SegFormer model')
    parser.add_argument('--test_dir', type=str, default='data/testImages',
                       help='Directory containing test images')
    parser.add_argument('--has_masks', action='store_true',
                       help='Whether test directory contains masks for evaluation')
    
    args = parser.parse_args()
    test_model(args.test_dir, args.has_masks)