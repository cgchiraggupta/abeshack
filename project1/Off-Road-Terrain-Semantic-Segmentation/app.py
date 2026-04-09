import streamlit as st
import torch
import cv2
import numpy as np
import os
from PIL import Image
from models.deeplabv3plus import get_model
import albumentations as A

# ---------------- CONFIG ----------------
MODEL_PATH = "checkpoints/best_model.pth"
IMG_SIZE = 512
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping: 0:Trees, 1:Lush Bushes, 2:Dry Grass, 3:Dry Bushes, 4:Ground Clutter, 
# 5:Flowers, 6:Logs, 7:Rocks, 8:Landscape, 9:Sky
COLORS = np.array([
    [34, 139, 34],    # 0: Trees
    [0, 255, 0],      # 1: Lush Bushes
    [189, 183, 107],  # 2: Dry Grass
    [160, 82, 45],    # 3: Dry Bushes
    [105, 105, 105],  # 4: Ground Clutter
    [255, 0, 255],    # 5: Flowers
    [139, 69, 19],    # 6: Logs
    [128, 128, 128],  # 7: Rocks
    [210, 180, 140],  # 8: Landscape (Catch-all)
    [135, 206, 235],  # 9: Sky
], dtype=np.uint8)

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
    "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# ---------------- FUNCTIONS ----------------

@st.cache_resource
def load_segmentation_model():
    model = get_model(NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def process_image(image, model, threshold=0.5):
    # Convert PIL to RGB numpy
    image_rgb = np.array(image.convert("RGB"))
    
    # Preprocessing
    transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
    aug = transform(image=image_rgb)
    img_resized = aug['image']
    
    # To Tensor
    input_tensor = img_resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
            output = model(input_tensor)
            # Multiclass softmax
            probs = torch.softmax(output, dim=1)
            
            # For simplicity in this demo, we'll use argmax for class assignment
            # but we could apply the threshold to the confidence of the winning class
            conf, pred = torch.max(probs, dim=1)
            pred = pred.squeeze().cpu().numpy()
            conf = conf.squeeze().cpu().numpy()
            
            # Apply threshold: everything below threshold becomes "Landscape" (index 8) or Background
            # Actually, in multiclass, thresholding is usually on the confidence
            mask = pred.copy()
            mask[conf < threshold] = 8 # Default to Landscape if low confidence

    # Colorize
    color_mask = COLORS[mask]
    
    # Overlay
    overlay = cv2.addWeighted(img_resized, 0.6, color_mask, 0.4, 0)
    
    return img_resized, color_mask, overlay, conf.mean()

# ---------------- UI ----------------

st.set_page_config(page_title="Offroad Terrain Segmenter", layout="wide")

st.title("🚙 Offroad Terrain Segmentation")
st.markdown("""
Upload an image of an off-road environment to see the model segment different terrain types!
This tool helps in identifying navigable paths and obstacles for off-road vehicles.
""")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_mask = st.checkbox("Show Mask Overlay", value=True)
    show_raw_mask = st.checkbox("Show Raw Predicted Mask", value=True)

uploaded_file = st.sidebar.file_uploader("Upload an offroad image", type=["jpg", "png", "jpeg"])

model = load_segmentation_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner("Running segmentation..."):
        img_resized, color_mask, overlay, mean_conf = process_image(image, model, threshold)

    st.success(f"Segmentation Complete! Mean Confidence: {mean_conf:.2%}")

    # Class Distribution statistics
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    st.subheader("Class Distribution")
    dist_cols = st.columns(5)
    
    sorted_indices = np.argsort(-counts) # Descending order
    
    for i, idx in enumerate(sorted_indices):
        class_id = unique[idx]
        percentage = (counts[idx] / total_pixels) * 100
        
        # Only show significant classes (>0.1%)
        if percentage > 0.1:
            class_name = CLASS_NAMES[class_id]
            color = COLORS[class_id]
            with dist_cols[i % 5]:
                 st.metric(
                    label=class_name, 
                    value=f"{percentage:.1f}%",
                    delta_color="off"
                 )

    # Display results
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("Original Image")
        st.image(img_resized, use_container_width=True)
    
    with c2:
        if show_raw_mask:
            st.subheader("Predicted Mask")
            st.image(color_mask, use_container_width=True)
        else:
            st.info("Raw mask hidden via settings.")

    with c3:
        if show_mask:
            st.subheader("Overlay Result")
            st.image(overlay, use_container_width=True)
        else:
            st.info("Overlay hidden via settings.")

    # Legend
    st.divider()
    st.subheader("Legend")
    l_cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        with l_cols[i % 5]:
            # Create a small color patch
            color = COLORS[i]
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                f'<div style="width: 20px; height: 20px; background-color: rgb({color[0]},{color[1]},{color[2]}); margin-right: 10px; border: 1px solid #fff;"></div>'
                f'<span>{name}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

else:
    st.info("Please upload an image from the sidebar to get started.")
    
    # Show sample images if they exist
    sample_dir = "data/testImages/Color_Images"
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]
        if samples:
            st.write("Or try one of these samples from the dataset:")
            cols = st.columns(len(samples))
            for i, s in enumerate(samples):
                with cols[i]:
                    img = Image.open(os.path.join(sample_dir, s))
                    st.image(img, caption=s, use_container_width=True)
