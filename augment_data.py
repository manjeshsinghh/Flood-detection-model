"""
Script to augment non-flood prone images to balance the dataset.
"""
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import random


def augment_images(
    input_dir: str,
    output_dir: str,
    target_count: int,
    image_size: int = 224
):
    """
    Augment images in input_dir to reach target_count images.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save augmented images
        target_count: Target number of images to generate
        image_size: Size for image resizing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    image_files = [f for f in input_path.rglob('*') if f.suffix in image_extensions and f.is_file()]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} source images in {input_dir}")
    print(f"Target: Generate {target_count} total images")
    
    # Count existing images in output dir
    existing_count = len([f for f in output_path.glob('*') if f.suffix.lower() in image_extensions])
    images_needed = target_count - existing_count
    
    if images_needed <= 0:
        print(f"Output directory already has {existing_count} images (target: {target_count})")
        return
    
    print(f"Need to generate {images_needed} additional images")
    
    # Define augmentation transforms
    augmentation_transforms = [
        # Basic augmentations
        transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ]),
        # Combined augmentations
        transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15)),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size + 30, image_size + 30)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((image_size + 25, image_size + 25)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]),
    ]
    
    # Generate augmented images
    generated_count = 0
    transform_idx = 0
    
    with tqdm(total=images_needed, desc="Generating augmented images") as pbar:
        while generated_count < images_needed:
            # Select a random source image
            source_image_path = random.choice(image_files)
            
            try:
                # Load image
                image = Image.open(source_image_path).convert('RGB')
                
                # Apply random augmentation
                transform = random.choice(augmentation_transforms)
                augmented_tensor = transform(image)
                
                # Convert back to PIL Image and save
                augmented_image = transforms.ToPILImage()(augmented_tensor)
                
                # Generate unique filename
                base_name = source_image_path.stem
                suffix = source_image_path.suffix
                counter = transform_idx % len(augmentation_transforms)
                new_filename = f"{base_name}_aug_{counter}_{generated_count:04d}{suffix}"
                output_file = output_path / new_filename
                
                # Skip if file already exists
                if output_file.exists():
                    continue
                
                augmented_image.save(output_file, quality=95)
                generated_count += 1
                transform_idx += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing {source_image_path}: {e}")
                continue
    
    # Verify final count
    final_count = len([f for f in output_path.glob('*') if f.suffix.lower() in image_extensions])
    print(f"\n{'='*60}")
    print(f"Augmentation Complete!")
    print(f"{'='*60}")
    print(f"Source images: {len(image_files)}")
    print(f"Generated images: {generated_count}")
    print(f"Total images in output: {final_count}")
    print(f"Output directory: {output_dir}")


def main():
    """Main function to augment non-flood prone images."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment non-flood prone images')
    parser.add_argument('--input-dir', type=str, default='River Basin/non-flood prone',
                        help='Directory containing source images')
    parser.add_argument('--output-dir', type=str, default='River Basin/non-flood prone',
                        help='Directory to save augmented images (default: same as input)')
    parser.add_argument('--target-count', type=int, default=1534,
                        help='Target number of images to generate (default: match flood-prone count)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for augmentation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Data Augmentation for Non-Flood Prone Images")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target count: {args.target_count}")
    print("="*60)
    
    augment_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_count=args.target_count,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()

