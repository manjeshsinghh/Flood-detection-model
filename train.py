"""
Training script for flood classification model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from model import create_model, count_parameters
from data_loader import create_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1


def validate_epoch(model, val_loader, criterion, device):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=0.001,
    device='cuda',
    save_dir='checkpoints',
    model_name='flood_classifier'
):
    """Train the model and save checkpoints."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    
    print(f"Starting training on {device}")
    print(f"Model has {count_parameters(model):,} trainable parameters")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history
            }, os.path.join(save_dir, f'{model_name}_epoch_{epoch + 1}.pth'))
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation F1 Score: {best_val_f1:.4f}")
    print("=" * 60)
    
    # Plot training history
    plot_training_history(history, save_dir, model_name)
    
    return history


def plot_training_history(history, save_dir, model_name):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'))
    plt.close()
    print(f"Training history plot saved to {os.path.join(save_dir, f'{model_name}_training_history.png')}")


def main():
    parser = argparse.ArgumentParser(description='Train flood classification model')
    parser.add_argument('--flood-dir', type=str, default='River Basin/flood prone',
                        help='Directory containing flood-prone images')
    parser.add_argument('--non-flood-dir', type=str, default='River Basin/non-flood prone',
                        help='Directory containing non-flood-prone images')
    parser.add_argument('--model-type', type=str, default='basic_cnn',
                        choices=['basic_cnn', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet_v2'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for input')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--model-name', type=str, default='flood_classifier',
                        help='Name for saved model files')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        flood_dir=args.flood_dir,
        non_flood_dir=args.non_flood_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        num_classes=2,
        pretrained=not args.no_pretrained,
        dropout_rate=0.5
    )
    model = model.to(device)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model_name
    )


if __name__ == '__main__':
    main()

