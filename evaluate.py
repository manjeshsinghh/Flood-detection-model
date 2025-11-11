"""
Evaluation and inference script for flood classification model.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import create_model
from data_loader import get_transforms


def load_model(model_path, model_type='resnet18', device='cuda', pretrained=True):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model(
        model_type=model_type,
        num_classes=2,
        pretrained=pretrained,
        dropout_rate=0.5
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, test_loader, device, class_names=['Non-Flood Prone', 'Flood Prone']):
    """Evaluate the model on test data."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:   {recall_per_class[i]:.4f}")
        print(f"    F1 Score: {f1_per_class[i]:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def predict_image(model, image_path, device, image_size=224, class_names=['Non-Flood Prone', 'Flood Prone']):
    """Predict a single image."""
    # Load and preprocess image
    transform = get_transforms(image_size, augment=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    
    pred_class = pred.item()
    confidence = probs[0][pred_class].item()
    
    result = {
        'predicted_class': class_names[pred_class],
        'class_id': pred_class,
        'confidence': confidence,
        'probabilities': {
            class_names[0]: probs[0][0].item(),
            class_names[1]: probs[0][1].item()
        }
    }
    
    return result


def predict_batch(model, image_dir, device, image_size=224, class_names=['Non-Flood Prone', 'Flood Prone']):
    """Predict all images in a directory."""
    results = []
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        import glob
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return results
    
    print(f"Found {len(image_files)} images")
    
    for image_path in tqdm(image_files, desc="Predicting"):
        result = predict_image(model, image_path, device, image_size, class_names)
        if result:
            result['image_path'] = image_path
            results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate flood classification model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='basic_cnn',
                        choices=['basic_cnn', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet_v2'],
                        help='Model architecture')
    parser.add_argument('--flood-dir', type=str, default='River Basin/flood prone',
                        help='Directory containing flood images (for test set evaluation)')
    parser.add_argument('--non-flood-dir', type=str, default='River Basin/non-flood prone',
                        help='Directory containing non-flood images (for test set evaluation)')
    parser.add_argument('--evaluate-test-set', action='store_true',
                        help='Evaluate on test set split from data directories')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path to single image for prediction')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Path to directory of images for batch prediction')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for input')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, checkpoint = load_model(args.model_path, args.model_type, device)
    
    if 'val_acc' in checkpoint:
        print(f"Model checkpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"  Validation F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    
    class_names = ['Non-Flood Prone', 'Flood Prone']
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on test set if requested
    if args.evaluate_test_set:
        from data_loader import create_data_loaders
        print("Creating test data loader...")
        _, _, test_loader = create_data_loaders(
            flood_dir=args.flood_dir,
            non_flood_dir=args.non_flood_dir,
            batch_size=32,
            image_size=args.image_size
        )
        results = evaluate_model(model, test_loader, device, class_names)
        
        # Save confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names,
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        # Save detailed results
        import json
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump({
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
                'confusion_matrix': results['confusion_matrix'].tolist()
            }, f, indent=2)
    
    # Single image prediction
    if args.image_path:
        print(f"\nPredicting image: {args.image_path}")
        result = predict_image(model, args.image_path, device, args.image_size, class_names)
        if result:
            print(f"Predicted: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
    
    # Batch prediction
    if args.image_dir:
        print(f"\nPredicting images in: {args.image_dir}")
        results = predict_batch(model, args.image_dir, device, args.image_size, class_names)
        
        # Save results
        import json
        output_path = os.path.join(args.output_dir, 'batch_predictions.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} predictions to {output_path}")
        
        # Print summary
        flood_count = sum(1 for r in results if r['class_id'] == 1)
        non_flood_count = len(results) - flood_count
        print(f"\nSummary:")
        print(f"  Flood Prone: {flood_count} ({flood_count/len(results)*100:.1f}%)")
        print(f"  Non-Flood Prone: {non_flood_count} ({non_flood_count/len(results)*100:.1f}%)")


if __name__ == '__main__':
    main()

