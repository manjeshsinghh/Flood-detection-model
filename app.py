"""
Streamlit Dashboard for Flood Prone Area Detection
"""
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import time

from model import create_model
from data_loader import get_transforms

# Page configuration
st.set_page_config(
    page_title="Flood Detection Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .flood-prone {
        background-color: #ffcccc;
        border: 3px solid #ff0000;
    }
    .non-flood-prone {
        background-color: #ccffcc;
        border: 3px solid #00ff00;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(model_path, model_type='basic_cnn', device='cpu', pretrained=True):
    """Load the trained model with caching."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Only use pretrained for ResNet, VGG, MobileNet models
        use_pretrained = pretrained if model_type not in ['basic_cnn', 'simple_cnn'] else False
        
        model = create_model(
            model_type=model_type,
            num_classes=2,
            pretrained=use_pretrained,
            dropout_rate=0.5
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("💡 Tip: Make sure you selected the correct model architecture type.")
        return None, None

def predict_image(model, image, device, image_size=224, class_names=['Non-Flood Prone', 'Flood Prone']):
    """Predict a single image."""
    # Get transform
    transform = get_transforms(image_size, augment=False)
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
    
    # Preprocess
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

def extract_frames_from_video(video_file, max_frames=10, frame_interval=30):
    """Extract frames from video file."""
    frames = []
    tfile = None
    
    try:
        # Determine file extension
        file_extension = os.path.splitext(video_file.name)[1] or '.mp4'
        
        # Save uploaded file to temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tfile.write(video_file.read())
        tfile.close()
        
        # Open video
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_count = 0
        extracted_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust frame_interval if video is too short
        if total_frames < frame_interval:
            frame_interval = max(1, total_frames // max_frames)
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
    finally:
        # Clean up temporary file
        if tfile and os.path.exists(tfile.name):
            try:
                os.unlink(tfile.name)
            except:
                pass
    
    return frames

# Sidebar for model configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Model selection
    model_path_options = {
        'Best Model': 'checkpoints/flood_classifier_best.pth',
        'Basic CNN Best': 'checkpoints/basic_flood_classifier_best.pth',
    }
    
    # Check which models exist
    available_models = {}
    for name, path in model_path_options.items():
        if os.path.exists(path):
            available_models[name] = path
    
    if not available_models:
        st.error("No model checkpoints found! Please train a model first.")
        st.stop()
    
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(available_models.keys())
    )
    model_path = available_models[selected_model_name]
    
    # Model type selection
    model_type = st.selectbox(
        "Model Architecture",
        options=['basic_cnn', 'simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet_v2'],
        index=0,
        help="Select the model architecture used during training"
    )
    
    # Load model button
    if st.button("🔄 Load Model", type="primary"):
        with st.spinner("Loading model..."):
            model, checkpoint = load_model(
                model_path,
                model_type=model_type,
                device=st.session_state.device,
                pretrained=True
            )
            
            if model is not None:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                
                # Display model info
                if checkpoint:
                    if 'epoch' in checkpoint:
                        st.info(f"**Epoch:** {checkpoint.get('epoch', 'N/A')}")
                    if 'val_acc' in checkpoint:
                        st.info(f"**Validation Accuracy:** {checkpoint.get('val_acc', 'N/A'):.4f}")
            else:
                st.error("Failed to load model. Please check the model type.")
    
    # Display device info
    st.info(f"**Device:** {st.session_state.device}")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Select and load a model
    2. Upload an image or video
    3. View predictions
    """)

# Main content
st.markdown('<h1 class="main-header">🌊 Flood Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Upload an image or video to detect if it's a flood-prone area")

# Check if model is loaded
if not st.session_state.model_loaded:
    st.warning("⚠️ Please load a model from the sidebar first.")
    st.stop()

# Tabs for image and video
tab1, tab2 = st.tabs(["📸 Image Upload", "🎥 Video Upload"])

# Image Upload Tab
with tab1:
    st.header("Image Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to predict if it's flood-prone"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Predict button
            if st.button("🔍 Predict", type="primary", use_container_width=True):
                with st.spinner("Predicting..."):
                    result = predict_image(
                        st.session_state.model,
                        image,
                        st.session_state.device,
                        image_size=224,
                        class_names=['Non-Flood Prone', 'Flood Prone']
                    )
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    
                    # Prediction box
                    if result['class_id'] == 1:  # Flood Prone
                        st.markdown(
                            f'<div class="prediction-box flood-prone">',
                            unsafe_allow_html=True
                        )
                        st.markdown(f"### 🌊 **Flood Prone Area**")
                    else:  # Non-Flood Prone
                        st.markdown(
                            f'<div class="prediction-box non-flood-prone">',
                            unsafe_allow_html=True
                        )
                        st.markdown(f"### ✅ **Non-Flood Prone Area**")
                    
                    # Confidence score
                    confidence_percent = result['confidence'] * 100
                    st.markdown(f"**Confidence:** {confidence_percent:.2f}%")
                    st.progress(confidence_percent / 100)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Probabilities
                    st.markdown("### Probability Distribution")
                    prob_data = result['probabilities']
                    
                    for class_name, prob in prob_data.items():
                        prob_percent = prob * 100
                        st.write(f"**{class_name}:** {prob_percent:.2f}%")
                        st.progress(prob_percent / 100)

# Video Upload Tab
with tab2:
    st.header("Video Prediction")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to predict flood-prone areas in frames"
    )
    
    if uploaded_video is not None:
        # Video settings
        col1, col2 = st.columns(2)
        with col1:
            max_frames = st.slider("Maximum frames to extract", 5, 30, 10)
        with col2:
            frame_interval = st.slider("Frame interval", 10, 60, 30)
        
        # Extract frames button
        if st.button("🎬 Extract and Predict Frames", type="primary"):
            with st.spinner("Extracting frames from video..."):
                frames = extract_frames_from_video(
                    uploaded_video,
                    max_frames=max_frames,
                    frame_interval=frame_interval
                )
                
                if not frames:
                    st.error("No frames extracted from video. Please try a different video.")
                else:
                    st.success(f"✅ Extracted {len(frames)} frames")
                    
                    # Predict each frame
                    st.markdown("### Frame Predictions")
                    
                    # Create grid layout for frames
                    cols_per_row = 3
                    predictions = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, frame in enumerate(frames):
                        # Update progress
                        progress = (i + 1) / len(frames)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {i+1} of {len(frames)}...")
                        
                        # Predict frame
                        result = predict_image(
                            st.session_state.model,
                            frame,
                            st.session_state.device,
                            image_size=224,
                            class_names=['Non-Flood Prone', 'Flood Prone']
                        )
                        predictions.append(result)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display frames and predictions in grid
                    for row_start in range(0, len(frames), cols_per_row):
                        cols = st.columns(cols_per_row)
                        row_frames = frames[row_start:row_start + cols_per_row]
                        row_predictions = predictions[row_start:row_start + cols_per_row]
                        
                        for col_idx, (frame, result) in enumerate(zip(row_frames, row_predictions)):
                            with cols[col_idx]:
                                # Display frame
                                st.image(frame, caption=f"Frame {row_start + col_idx + 1}", use_container_width=True)
                                
                                # Display prediction
                                if result['class_id'] == 1:
                                    st.markdown("🌊 **Flood Prone**")
                                    st.markdown(f'<span style="color: red;">Confidence: {result["confidence"]*100:.1f}%</span>', unsafe_allow_html=True)
                                else:
                                    st.markdown("✅ **Non-Flood Prone**")
                                    st.markdown(f'<span style="color: green;">Confidence: {result["confidence"]*100:.1f}%</span>', unsafe_allow_html=True)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.markdown("### 📊 Video Summary")
                    
                    flood_count = sum(1 for p in predictions if p['class_id'] == 1)
                    non_flood_count = len(predictions) - flood_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Frames", len(predictions))
                    with col2:
                        st.metric("Flood Prone Frames", flood_count, f"{flood_count/len(predictions)*100:.1f}%")
                    with col3:
                        st.metric("Non-Flood Prone Frames", non_flood_count, f"{non_flood_count/len(predictions)*100:.1f}%")
                    
                    # Overall prediction
                    avg_flood_prob = np.mean([p['probabilities']['Flood Prone'] for p in predictions])
                    overall_prediction = "Flood Prone" if avg_flood_prob > 0.5 else "Non-Flood Prone"
                    
                    st.markdown("### 🎯 Overall Video Prediction")
                    if overall_prediction == "Flood Prone":
                        st.error(f"**{overall_prediction}** (Average confidence: {avg_flood_prob*100:.1f}%)")
                    else:
                        st.success(f"**{overall_prediction}** (Average confidence: {(1-avg_flood_prob)*100:.1f}%)")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Flood Detection Dashboard | Built with Streamlit</div>",
    unsafe_allow_html=True
)

