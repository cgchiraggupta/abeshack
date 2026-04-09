import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
import sys
sys.path.append('.')

from models.attention_unet import get_model

class TerrainSegmentationApp:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = get_model(num_classes=self.config['model']['num_classes'])
        self.model = self.model.to(self.device)
        
        checkpoint_path = self.config['app']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"Loaded model from {checkpoint_path}")
        else:
            st.warning(f"Checkpoint not found at {checkpoint_path}")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_colors = {
            0: [255, 0, 0],      # Trees: Red
            1: [0, 255, 0],      # Lush Bushes: Green
            2: [0, 128, 0],      # Dry Bushes: Dark Green
            3: [144, 238, 144],  # Grass: Light Green
            4: [139, 69, 19],    # Dirt: Brown
            5: [169, 169, 169],  # Gravel: Gray
            6: [105, 105, 105],  # Rocks: Dark Gray
            7: [255, 255, 0],    # Sand: Yellow
            8: [0, 0, 255],      # Water: Blue
            9: [135, 206, 235]   # Sky: Light Blue
        }
        
        self.class_names = {
            0: "Trees",
            1: "Lush Bushes", 
            2: "Dry Bushes",
            3: "Grass",
            4: "Dirt",
            5: "Gravel",
            6: "Rocks",
            7: "Sand",
            8: "Water",
            9: "Sky"
        }
    
    def preprocess_image(self, image):
        original_size = image.size
        image = image.resize((512, 512), Image.BILINEAR)
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor, original_size
    
    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1)
        return prediction.squeeze().cpu().numpy()
    
    def create_colored_mask(self, prediction):
        h, w = prediction.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            mask = prediction == class_idx
            colored_mask[mask] = color
        return colored_mask
    
    def create_overlay(self, original_image, colored_mask, alpha=0.5):
        original_np = np.array(original_image)
        overlay = original_np.copy()
        mask_indices = np.any(colored_mask != [0, 0, 0], axis=-1)
        overlay[mask_indices] = (alpha * colored_mask[mask_indices] + 
                                (1 - alpha) * original_np[mask_indices]).astype(np.uint8)
        return overlay
    
    def run(self):
        st.set_page_config(
            page_title="Off-Road Terrain Segmentation - Attention UNet",
            page_icon="🌲",
            layout="wide"
        )
        
        st.title("🌲 Off-Road Terrain Semantic Segmentation")
        st.markdown("### Attention UNet Model")
        st.markdown("Upload an off-road terrain image to get semantic segmentation predictions.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.sidebar.header("Settings")
            
            uploaded_file = st.sidebar.file_uploader(
                "Choose an image...", 
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            alpha = st.sidebar.slider(
                "Overlay transparency", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.1
            )
            
            show_legend = st.sidebar.checkbox("Show class legend", value=True)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Model Information")
            st.sidebar.info(f"**Model**: Attention UNet\n\n"
                          f"**Classes**: {self.config['model']['num_classes']}\n\n"
                          f"**Device**: {self.device}")
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### How to use")
            st.sidebar.markdown("1. Upload an off-road terrain image\n"
                              "2. Adjust overlay transparency if needed\n"
                              "3. View segmentation results")
        
        with col2:
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    with st.spinner('Processing image...'):
                        image_tensor, original_size = self.preprocess_image(image)
                        prediction = self.predict(image_tensor)
                        
                        prediction_resized = Image.fromarray(prediction.astype(np.uint8))
                        prediction_resized = prediction_resized.resize(original_size, Image.NEAREST)
                        prediction = np.array(prediction_resized)
                        
                        colored_mask = self.create_colored_mask(prediction)
                        overlay = self.create_overlay(image, colored_mask, alpha)
                    
                    st.success("Segmentation complete!")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Original", "Segmentation", "Colored Mask", "Overlay"])
                    
                    with tab1:
                        st.image(image, caption="Original Image", use_column_width=True)
                    
                    with tab2:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(prediction, cmap='tab20', vmin=0, vmax=self.config['model']['num_classes']-1)
                        ax.axis('off')
                        ax.set_title("Semantic Segmentation")
                        st.pyplot(fig)
                    
                    with tab3:
                        st.image(colored_mask, caption="Colored Mask", use_column_width=True)
                    
                    with tab4:
                        st.image(overlay, caption=f"Overlay (α={alpha})", use_column_width=True)
                    
                    st.markdown("---")
                    st.subheader("Class Distribution")
                    
                    class_counts = np.bincount(prediction.flatten(), minlength=self.config['model']['num_classes'])
                    total_pixels = prediction.size
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Pixel Counts:**")
                        for class_idx in range(self.config['model']['num_classes']):
                            if class_counts[class_idx] > 0:
                                percentage = (class_counts[class_idx] / total_pixels) * 100
                                st.markdown(f"**{self.class_names.get(class_idx, f'Class {class_idx}')}**: "
                                          f"{class_counts[class_idx]:,} pixels ({percentage:.1f}%)")
                    
                    with col_b:
                        if show_legend:
                            st.markdown("**Class Legend:**")
                            legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
                            for class_idx, class_name in self.class_names.items():
                                color = self.class_colors[class_idx]
                                color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                                legend_html += f"""
                                <div style='display: flex; align-items: center; margin: 5px;'>
                                    <div style='width: 20px; height: 20px; background-color: {color_hex}; margin-right: 8px; border: 1px solid #ccc;'></div>
                                    <span>{class_name}</span>
                                </div>
                                """
                            legend_html += "</div>"
                            st.markdown(legend_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("Download Results")
                    
                    col_x, col_y, col_z = st.columns(3)
                    
                    with col_x:
                        img_buffer = Image.fromarray(prediction.astype(np.uint8))
                        st.download_button(
                            label="Download Segmentation",
                            data=img_buffer.tobytes(),
                            file_name="segmentation.png",
                            mime="image/png"
                        )
                    
                    with col_y:
                        mask_buffer = Image.fromarray(colored_mask)
                        st.download_button(
                            label="Download Colored Mask",
                            data=mask_buffer.tobytes(),
                            file_name="colored_mask.png",
                            mime="image/png"
                        )
                    
                    with col_z:
                        overlay_buffer = Image.fromarray(overlay)
                        st.download_button(
                            label="Download Overlay",
                            data=overlay_buffer.tobytes(),
                            file_name="overlay.png",
                            mime="image/png"
                        )
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        
                        axes[0, 0].imshow(image)
                        axes[0, 0].set_title('Original Image')
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(prediction, cmap='tab20', vmin=0, vmax=self.config['model']['num_classes']-1)
                        axes[0, 1].set_title('Semantic Segmentation')
                        axes[0, 1].axis('off')
                        
                        axes[1, 0].imshow(colored_mask)
                        axes[1, 0].set_title('Colored Mask')
                        axes[1, 0].axis('off')
                        
                        axes[1, 1].imshow(overlay)
                        axes[1, 1].set_title(f'Overlay (α={alpha})')
                        axes[1, 1].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        with open(tmp_file.name, 'rb') as f:
                            st.download_button(
                                label="Download All Results (4-in-1)",
                                data=f.read(),
                                file_name="all_results.png",
                                mime="image/png"
                            )
                        
                        os.unlink(tmp_file.name)
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.exception(e)
            
            else:
                st.info("👈 Please upload an image using the file uploader in the sidebar.")
                
                st.markdown("---")
                st.subheader("About this Application")
                st.markdown("""
                This application uses an **Attention UNet** model for semantic segmentation of off-road terrain images.
                
                **Features:**
                - Semantic segmentation of 10 terrain classes
                - Interactive visualization of results
                - Class distribution analysis
                - Downloadable results
                
                **Supported Classes:**
                1. Trees 🌳
                2. Lush Bushes 🌿
                3. Dry Bushes 🍂
                4. Grass 🌱
                5. Dirt 🟤
                6. Gravel ⚫
                7. Rocks 🪨
                8. Sand 🟡
                9. Water 💧
                10. Sky ☁️
                """)
                
                st.markdown("---")
                st.subheader("Sample Images")
                
                sample_col1, sample_col2, sample_col3 = st.columns(3)
                
                with sample_col1:
                    st.image("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop", 
                            caption="Mountain Trail", use_column_width=True)
                
                with sample_col2:
                    st.image("https://images.unsplash.com/photo-1519681393784-d120267933ba?w-400&h=300&fit=crop", 
                            caption="Forest Path", use_column_width=True)
                
                with sample_col3:
                    st.image("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop", 
                            caption="Desert Terrain", use_column_width=True)

def main():
    app = TerrainSegmentationApp()
    app.run()

if __name__ == "__main__":
    main()