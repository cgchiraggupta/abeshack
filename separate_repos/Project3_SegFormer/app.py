import streamlit as st
import torch
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="Off-Road Terrain Segmentation - SegFormer",
    page_icon="🌲",
    layout="wide"
)

@st.cache_resource
def load_model():
    from models.segformer import get_model
    MODEL_PATH = "checkpoints/best_model.pth"
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, DEVICE

def preprocess_image(image, size=512):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    
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

def get_class_distribution(mask):
    unique, counts = np.unique(mask, return_counts=True)
    distribution = dict(zip(unique, counts))
    total_pixels = mask.size
    return distribution, total_pixels

def main():
    st.title("🌲 Off-Road Terrain Segmentation - SegFormer-B2")
    st.markdown("""
    This interactive dashboard allows you to segment off-road terrain images using SegFormer-B2 from HuggingFace.
    Upload an image to see the segmentation results.
    """)
    
    COLORS = np.array([
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
    
    CLASS_NAMES = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    
    sidebar = st.sidebar
    sidebar.header("Settings")
    
    confidence_threshold = sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for class prediction"
    )
    
    overlay_alpha = sidebar.slider(
        "Overlay Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Transparency of mask overlay on original image"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an off-road terrain image for segmentation"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            use_sample = st.checkbox("Use sample image instead", value=False)
        else:
            use_sample = st.checkbox("Use sample image", value=True)
        
        if use_sample:
            sample_images = {
                "Sample 1": "assets/sample1.jpg",
                "Sample 2": "assets/sample2.jpg",
                "Sample 3": "assets/sample3.jpg"
            }
            
            sample_choice = st.selectbox("Select sample image", list(sample_images.keys()))
            
            sample_path = sample_images[sample_choice]
            if os.path.exists(sample_path):
                image = Image.open(sample_path)
                st.image(image, caption=f"Sample Image: {sample_choice}", use_column_width=True)
            else:
                st.warning("Sample images not found. Please upload your own image.")
                return
    
    with col2:
        if 'image' in locals():
            st.subheader("Model Information")
            
            st.markdown("""
            **Selected Model:** SegFormer-B2 (HuggingFace Transformers)
            - **Architecture:** Transformer-based segmentation with hierarchical encoder
            - **Backbone:** Mix Transformer (MiT-B2)
            - **Parameters:** ~27 million
            - **Strengths:** Excellent at capturing global context, state-of-the-art on many benchmarks
            - **Best For:** Complex scenes requiring global understanding
            """)
            
            if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
                with st.spinner("Running SegFormer-B2 segmentation..."):
                    start_time = time.time()
                    
                    try:
                        model, device = load_model()
                        
                        image_tensor, original_h, original_w, image_resized = preprocess_image(image)
                        image_tensor = image_tensor.to(device)
                        
                        with torch.no_grad():
                            output = model(image_tensor)
                        
                        pred_mask = postprocess_mask(output, original_h, original_w)
                        
                        color_mask = create_color_mask(pred_mask, COLORS)
                        overlay = create_overlay(np.array(image), color_mask, overlay_alpha)
                        
                        inference_time = time.time() - start_time
                        
                        st.success(f"Segmentation completed in {inference_time:.2f} seconds!")
                        
                        distribution, total_pixels = get_class_distribution(pred_mask)
                        
                        st.markdown("---")
                        st.subheader("Segmentation Results")
                        
                        results_col1, results_col2, results_col3 = st.columns(3)
                        
                        with results_col1:
                            st.image(image, caption="Original Image", use_column_width=True)
                        
                        with results_col2:
                            st.image(color_mask, caption="Predicted Mask", use_column_width=True)
                        
                        with results_col3:
                            st.image(overlay, caption="Overlay", use_column_width=True)
                        
                        st.markdown("---")
                        st.subheader("Class Distribution")
                        
                        dist_col1, dist_col2 = st.columns([2, 1])
                        
                        with dist_col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            class_counts = []
                            for class_id in range(10):
                                count = distribution.get(class_id, 0)
                                percentage = (count / total_pixels) * 100
                                class_counts.append(percentage)
                            
                            bars = ax.bar(CLASS_NAMES, class_counts)
                            ax.set_xlabel('Class')
                            ax.set_ylabel('Percentage (%)')
                            ax.set_title('Class Distribution in Prediction')
                            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
                            
                            for bar, count in zip(bars, class_counts):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                       f'{count:.1f}%', ha='center', va='bottom', fontsize=8)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with dist_col2:
                            st.markdown("**Class Legend:**")
                            for i, (color, name) in enumerate(zip(COLORS, CLASS_NAMES)):
                                color_hex = '#%02x%02x%02x' % tuple(color)
                                count = distribution.get(i, 0)
                                percentage = (count / total_pixels) * 100
                                st.markdown(
                                    f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                                    f'<div style="width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px; border: 1px solid #ccc;"></div>'
                                    f'<span>{name}: {percentage:.1f}%</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        
                        st.markdown("---")
                        st.subheader("Export Results")
                        
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            if st.button("💾 Save Mask", use_container_width=True):
                                mask_img = Image.fromarray(color_mask)
                                mask_img.save("predicted_mask_segformer.png")
                                st.success("Mask saved as predicted_mask_segformer.png!")
                        
                        with export_col2:
                            if st.button("📊 Save Distribution", use_container_width=True):
                                with open("distribution_segformer.txt", "w") as f:
                                    f.write(f"Model: SegFormer-B2\n")
                                    f.write(f"Image: {uploaded_file.name if uploaded_file else sample_choice}\n")
                                    f.write(f"Total Pixels: {total_pixels}\n\n")
                                    f.write("Class Distribution:\n")
                                    for i, name in enumerate(CLASS_NAMES):
                                        count = distribution.get(i, 0)
                                        percentage = (count / total_pixels) * 100
                                        f.write(f"{name}: {count} pixels ({percentage:.2f}%)\n")
                                st.success("Distribution saved as distribution_segformer.txt!")
                        
                        with export_col3:
                            if st.button("🖼️ Save Overlay", use_container_width=True):
                                overlay_img = Image.fromarray(overlay)
                                overlay_img.save("overlay_segformer.png")
                                st.success("Overlay saved as overlay_segformer.png!")
                    
                    except Exception as e:
                        st.error(f"Error during segmentation: {str(e)}")
                        st.exception(e)
    
    st.markdown("---")
    st.markdown("""
    ### About This Application
    
    This application uses SegFormer-B2 (Transformer-based) to segment off-road terrain images into 10 classes:
    
    1. **Trees** - Forest areas and individual trees
    2. **Lush Bushes** - Green, healthy vegetation
    3. **Dry Grass** - Brown, dry grassy areas
    4. **Dry Bushes** - Brown, dry bushes
    5. **Ground Clutter** - Miscellaneous ground debris
    6. **Flowers** - Flowering plants
    7. **Logs** - Fallen trees and logs
    8. **Rocks** - Rock formations and boulders
    9. **Landscape** - General terrain and background
    10. **Sky** - Sky areas
    
    **Model Details:**
    - **Architecture:** SegFormer-B2 with Mix Transformer encoder
    - **Training:** 12 epochs with early stopping
    - **Best Validation Dice Score:** 0.63
    - **Loss Function:** Combined (CrossEntropy + Tversky + Focal)
    - **Optimization:** Mixed Precision Training (AMP)
    
    Upload your own off-road image or use the sample images to test the segmentation!
    """)

if __name__ == "__main__":
    main()