# Flood Detection Dashboard

A deep learning application that classifies river basin images as **flood-prone** or **non-flood-prone** using CNN models. Includes an interactive Streamlit dashboard for easy image and video predictions.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser. Upload images or videos to get predictions!

## Features

- 🌊 **Image Classification**: Upload images to detect flood-prone areas
- 🎥 **Video Analysis**: Upload videos to analyze multiple frames
- 📊 **Interactive Dashboard**: User-friendly web interface
- 🤖 **Multiple Models**: Support for various CNN architectures
- 📈 **Confidence Scores**: Get detailed prediction probabilities

## Usage

### Streamlit Dashboard

1. Select a model checkpoint from the sidebar
2. Choose the model architecture (e.g., `basic_cnn`, `resnet18`)
3. Click "Load Model"
4. Upload an image or video file
5. Click "Predict" to see results

### Training a Model

```bash
python train.py --model-type basic_cnn --epochs 50
```

### Command Line Prediction

```bash
python evaluate.py \
    --model-path checkpoints/flood_classifier_best.pth \
    --model-type basic_cnn \
    --image-path path/to/image.jpg
```

## Project Structure

```
.
├── app.py              # Streamlit dashboard
├── model.py            # Model architectures
├── data_loader.py      # Data loading utilities
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── augment_data.py     # Data augmentation script
├── checkpoints/        # Saved model checkpoints
└── River Basin/        # Dataset (flood prone & non-flood prone)
```

## Model Types

- `basic_cnn` - Lightweight CNN (default)
- `simple_cnn` - CNN with batch normalization
- `resnet18`, `resnet34`, `resnet50` - Transfer learning models
- `vgg16`, `mobilenet_v2` - Other architectures

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Streamlit
- OpenCV (for video processing)

See `requirements.txt` for complete dependencies.

## Data

The dataset contains:
- **Flood-prone images**: 1,534 images
- **Non-flood-prone images**: 1,534 images (augmented)

Images are located in `River Basin/flood prone/` and `River Basin/non-flood prone/`.

## License

This project is provided for research and educational purposes.
