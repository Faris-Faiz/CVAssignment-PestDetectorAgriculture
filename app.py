import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import supervision as sv
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

# Set page config
st.set_page_config(
    page_title="YOLOv11 + SAM2.1 Segmentation", 
    page_icon="🖼️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model information
MODEL_INFO = {
    'yolo': {
        'filename': 'pest_detector_YOLOv11_Model.pt',
        'download_link': 'https://drive.google.com/file/d/1ER_oKZ9ZiPImReSPlUY8E5rdm7bSZto6/view?usp=sharing'
    },
    'sam_config': {
        'filename': 'sam2.1_hiera_b+.yaml',
        'download_link': 'https://drive.google.com/file/d/1ndNQmTTTjeW3x-Tnt_9paAVzIhG4lgf7/view?usp=sharing'
    },
    'sam_checkpoint': {
        'filename': 'checkpoint-04012025-0103(CV_Assignment).pt',
        'download_link': 'https://drive.google.com/file/d/1wLeEyCj49T7ibyOAqoECz7Qr8zPlnVSy/view?usp=sharing'
    }
}

def check_models_exist():
    """Check if required models exist in the models directory"""
    models_dir = os.path.join(os.getcwd(), 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return False
    
    required_files = [
        os.path.join(models_dir, MODEL_INFO['yolo']['filename']),
        os.path.join(models_dir, MODEL_INFO['sam_config']['filename']),
        os.path.join(models_dir, MODEL_INFO['sam_checkpoint']['filename'])
    ]
    
    return all(os.path.exists(file) for file in required_files)

def show_download_instructions():
    """Display instructions for downloading required models"""
    st.error("Required models not found. Please download the following models and place them in the 'models' directory:")
    
    for model_type, info in MODEL_INFO.items():
        st.markdown(f"""
        * Download [{info['filename']}]({info['download_link']})
        """)
    
    st.info("After downloading, place the files in the 'models' directory and refresh the page.")

def load_models():
    """Load YOLO and SAM2.1 models"""
    try:
        models_dir = os.path.join(os.getcwd(), 'models')
        yolo_model_path = os.path.join(models_dir, MODEL_INFO['yolo']['filename'])
        sam21_cfg_path = os.path.join(models_dir, MODEL_INFO['sam_config']['filename'])
        sam21_checkpoint_path = os.path.join(models_dir, MODEL_INFO['sam_checkpoint']['filename'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        yolo_model = YOLO(yolo_model_path).to(device)
        sam2_model = build_sam2(sam21_cfg_path, sam21_checkpoint_path, device=str(device))
        predictor = SAM2ImagePredictor(sam2_model)

        return yolo_model, predictor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def process_image(image, yolo_model, sam_predictor):
    """Process image through YOLO and SAM2.1 models"""
    try:
        img_array = np.array(image)
        result = yolo_model.predict(source=img_array, conf=0.25)[0]
        yolo_detections = sv.Detections.from_ultralytics(result)
        
        if len(yolo_detections) == 0:
            return None, None, "No objects detected in the image."

        boxes_np = yolo_detections.xyxy
        sam_predictor.set_image(img_array)
        masks, _, _ = sam_predictor.predict(box=boxes_np, multimask_output=False)
        masks_boolean = np.squeeze(masks).astype(bool)
        
        if len(masks_boolean.shape) == 2:
            masks_boolean = masks_boolean[np.newaxis, ...]

        detections = sv.Detections(
            mask=masks_boolean,
            xyxy=boxes_np,
            class_id=yolo_detections.class_id,
            confidence=yolo_detections.confidence
        )

        return detections, masks_boolean, None
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

def create_visualization(image, detections):
    """Create visualization of the segmentation results"""
    try:
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image = np.array(image).copy()
        annotated_image = mask_annotator.annotate(
            scene=annotated_image,
            detections=detections
        )
        return annotated_image
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Main app
st.title("YOLOv11 + SAM2.1 Combined Segmentation")
st.markdown("""
This application combines YOLOv11 object detection with SAM2.1 segmentation to precisely identify and segment objects in images.
""")

# Check if models exist
if not check_models_exist():
    show_download_instructions()
    st.stop()

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load models
if not st.session_state.models_loaded:
    with st.spinner("Loading models... This may take a moment."):
        yolo_model, sam_predictor = load_models()
        if yolo_model is not None and sam_predictor is not None:
            st.session_state.yolo_model = yolo_model
            st.session_state.sam_predictor = sam_predictor
            st.session_state.models_loaded = True
        else:
            st.error("Failed to load models. Please refresh the page to try again.")
            st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            detections, masks, error = process_image(
                image, 
                st.session_state.yolo_model, 
                st.session_state.sam_predictor
            )
            
            if error:
                st.error(error)
            elif detections is not None:
                st.session_state.masks = masks
                annotated_image = create_visualization(image, detections)
                
                if annotated_image is not None:
                    st.write("### Segmentation Results")
                    st.image(annotated_image, caption="Segmented Image", use_container_width=True)
                    
                    # Save results
                    buf = io.BytesIO()
                    Image.fromarray(annotated_image).save(buf, format="PNG")
                    buf.seek(0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download Segmented Image",
                            data=buf,
                            file_name="segmented_image.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                        mask_buf = io.BytesIO()
                        Image.fromarray(combined_mask).save(mask_buf, format="PNG")
                        mask_buf.seek(0)
                        
                        st.download_button(
                            label="Download Combined Mask",
                            data=mask_buf,
                            file_name="combined_mask.png",
                            mime="image/png"
                        )
else:
    st.info("Please upload an image to begin segmentation.")

# Footer
st.markdown("---")
st.markdown("Developed by Faris Faiz | [GitHub](https://github.com/Faris-Faiz)")
