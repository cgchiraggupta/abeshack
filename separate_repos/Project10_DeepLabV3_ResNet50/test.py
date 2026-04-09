import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Custom imports
from dataset.dataset import OffRoadDataset, get_transforms
from models.deeplabv3_resnet50 import DeepLabV3ResNet50
from metrics import dice_score, iou_score, SegmentationMetrics

def test_model(model_path, data_dir, device='cuda', batch_size=4, save_results=True):
    """
    Test the trained model on test set
    """
    print("=" * 60)
    print("Testing DeepLabV3+ ResNet50 Model")
    print("=" * 60)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = DeepLabV3ResNet50(num_classes=10)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with strict=False...")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    model.to(device)
    model.eval()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create test dataset and dataloader
    print("\nLoading test dataset...")
    test_transform = get_transforms('test', image_size=512)
    
    test_dataset = OffRoadDataset(
        data_dir=data_dir,
        split='test',
        transform=test_transform,
        debug=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Initialize metrics
    print("\nStarting testing...")
    metrics_calc = SegmentationMetrics(num_classes=10, device=device)
    
    # For confusion matrix
    all_predictions = []
    all_targets = []
    
    # For timing
    inference_times = []
    
    # Test loop
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing', leave=True)
        
        for batch_idx, (images, masks, paths) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            outputs = model(images)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics calculator
            metrics_calc.update(preds, masks)
            
            # Store for confusion matrix
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            
            # Update progress bar
            batch_dice = dice_score(preds, masks).item()
            batch_iou = iou_score(preds, masks).item()
            
            progress_bar.set_postfix({
                'dice': f'{batch_dice:.4f}',
                'iou': f'{batch_iou:.4f}',
                'time': f'{inference_time*1000:.1f}ms'
            })
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_dict = metrics_calc.compute()
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(10))
    
    # Calculate per-class metrics from confusion matrix
    class_names = [
        'Trees', 'Lush Bushes', 'Grass', 'Dirt', 'Sand',
        'Water', 'Rocks', 'Bushes', 'Mud', 'Background'
    ]
    
    per_class_iou = {}
    per_class_dice = {}
    
    for i in range(10):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-8)
        per_class_iou[class_names[i]] = iou
        
        # Dice
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        per_class_dice[class_names[i]] = dice
    
    # Calculate inference statistics
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    std_inference_time = np.std(inference_times) * 1000
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Compile results
    test_results = {
        'model': 'DeepLabV3+ ResNet50',
        'model_path': model_path,
        'test_samples': len(test_dataset),
        'batch_size': batch_size,
        'device': str(device),
        
        # Overall metrics
        'dice_score': metrics_dict.get('dice', 0),
        'iou_score': metrics_dict.get('iou', 0),
        'precision': metrics_dict.get('precision', 0),
        'recall': metrics_dict.get('recall', 0),
        'f1_score': metrics_dict.get('f1', 0),
        'accuracy': metrics_dict.get('accuracy', 0),
        'mean_iou': metrics_dict.get('mean_iou', 0),
        'mean_dice': metrics_dict.get('mean_dice', 0),
        
        # Per-class metrics
        'per_class_iou': per_class_iou,
        'per_class_dice': per_class_dice,
        
        # Inference performance
        'avg_inference_time_ms': avg_inference_time,
        'std_inference_time_ms': std_inference_time,
        'fps': fps,
        'total_inference_time_s': sum(inference_times),
        
        # Model info
        'parameters': total_params,
        'input_size': 512
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"Dice Score: {test_results['dice_score']:.4f}")
    print(f"IoU Score: {test_results['iou_score']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1 Score: {test_results['f1_score']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Mean IoU: {test_results['mean_iou']:.4f}")
    print(f"Mean Dice: {test_results['mean_dice']:.4f}")
    
    print(f"\nInference Performance:")
    print(f"Average inference time: {avg_inference_time:.1f} ± {std_inference_time:.1f} ms")
    print(f"FPS: {fps:.1f}")
    print(f"Total inference time: {test_results['total_inference_time_s']:.2f} s")
    
    print(f"\nPer-class IoU Scores:")
    for class_name, iou in per_class_iou.items():
        if class_name != 'Background':  # Skip background for cleaner output
            print(f"  {class_name}: {iou:.4f}")
    
    # Save results if requested
    if save_results:
        # Create results directory
        results_dir = 'test_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results as JSON
        results_path = os.path.join(results_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to: {results_path}")
        
        # Save confusion matrix plot
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - DeepLabV3+ ResNet50')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Create visualization of per-class metrics
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Per-class IoU
        classes = list(per_class_iou.keys())
        iou_values = list(per_class_iou.values())
        
        axes[0].bar(classes, iou_values, color='skyblue')
        axes[0].set_title('Per-class IoU Scores')
        axes[0].set_ylabel('IoU Score')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Per-class Dice
        dice_values = list(per_class_dice.values())
        
        axes[1].bar(classes, dice_values, color='lightgreen')
        axes[1].set_title('Per-class Dice Scores')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('DeepLabV3+ ResNet50 - Test Results', fontsize=16)
        plt.tight_layout()
        
        analysis_path = os.path.join(results_dir, 'results_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results analysis plot saved to: {analysis_path}")
    
    return test_results

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DeepLabV3+ ResNet50 model')
    parser.add_argument('--model', type=str, default='best_model.pth', 
                       help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory of dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save test results to files')
    
    args = parser.parse_args()
    
    # Test model
    results = test_model(
        model_path=args.model,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        save_results=args.save_results
    )
    
    return results

if __name__ == "__main__":
    main()