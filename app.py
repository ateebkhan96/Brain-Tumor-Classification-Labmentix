import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #4caf50;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 0.25rem;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 20px;
        background-color: #4caf50;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom CNN model
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create ResNet50 model
def create_resnet50_model(num_classes=4):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Preprocessing
def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((int(input_size * 1.14), int(input_size * 1.14))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Load model
@st.cache_resource
def load_model(model_type):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Updated model paths to match your GitHub repository
        if model_type == "ResNet50":
            model_path = "models/best_ResNet50.pth"
        else:
            model_path = "models/best_CustomCNN.pth"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please ensure the model files are in the correct directory.")
            return None, None, None, None

        checkpoint = torch.load(model_path, map_location=device)
        saved_class_names = checkpoint.get('class_names', ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
        input_size = checkpoint.get('input_size', 224)

        if model_type == "ResNet50":
            model = create_resnet50_model(num_classes=len(saved_class_names))
        else:
            model = BrainTumorCNN(num_classes=len(saved_class_names))

        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.to(device)
        model.eval()

        return model, device, saved_class_names, input_size
    except Exception as e:
        st.error(f"Error loading model from disk: {str(e)}")
        return None, None, None, None

# Predict
def predict_image(model, image, device, class_names, input_size=224):
    try:
        transform = get_transform(input_size)
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), probs.squeeze().cpu().numpy()
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an MRI image to classify brain tumors using deep learning models")

    with st.sidebar:
        st.header("🔧 Model Configuration")
        model_type = st.selectbox("Select Model", ["ResNet50", "Custom CNN"])
        
        # Updated info display
        if model_type == "ResNet50":
            st.info(f"📁 Using model: `models/best_ResNet50.pth`")
        else:
            st.info(f"📁 Using model: `models/best_CustomCNN.pth`")

        st.markdown('<div class="model-info">', unsafe_allow_html=True)
        if model_type == "ResNet50":
            st.markdown("""
            **ResNet50 Model**
            - Pre-trained on ImageNet
            - Transfer learning approach
            - Higher accuracy (92.63%)
            - Larger model size
            """)
        else:
            st.markdown("""
            **Custom CNN Model**
            - Built from scratch
            - Lightweight architecture
            - Good accuracy (88.21%)
            - Faster inference
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app classifies brain MRI scans into four categories:
        - **Glioma**
        - **Meningioma**
        - **No Tumor**
        - **Pituitary**
        """)
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ This is not a medical diagnosis tool. Consult professionals for real evaluations.")
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            st.markdown(f"""
            **Image Details:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {uploaded_file.type}
            """)

    with col2:
        st.subheader("🔍 Prediction Results")
        if uploaded_file is not None:
            with st.spinner("Loading model..."):
                model, device, class_names, input_size = load_model(model_type)

            if model is not None:
                st.success(f"✅ {model_type} model loaded successfully!")
                st.info(f"📊 Input size: {input_size}x{input_size}")
                st.info(f"🏷️ Classes: {', '.join(class_names)}")

                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, probabilities = predict_image(
                        model, image, device, class_names, input_size
                    )

                if predicted_class is not None:
                    predicted_label = class_names[predicted_class]
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Prediction: **{predicted_label}**")
                    st.markdown(f"### 📊 Confidence: **{confidence:.2%}**")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.subheader("📈 Class Probabilities")
                    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                        col_name, col_bar = st.columns([1, 2])
                        with col_name:
                            st.write(f"**{class_name}:**")
                        with col_bar:
                            bar_html = f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {prob*100:.1f}%;">
                                    {prob:.3f} ({prob*100:.1f}%)
                                </div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)

                    st.subheader("💡 Interpretation")
                    if predicted_label == "No Tumor":
                        st.success("✅ No tumor detected in this MRI scan.")
                    else:
                        st.info(f"🔍 Detected a **{predicted_label}** tumor with {confidence:.1%} confidence.")
                    if confidence > 0.9:
                        st.success("🎯 High confidence prediction")
                    elif confidence > 0.7:
                        st.warning("⚠️ Moderate confidence - consider more scans")
                    else:
                        st.error("❌ Low confidence - result may not be reliable")
            else:
                st.error("Failed to load model.")

        else:
            st.info("👆 Please upload an MRI image to begin")

    st.markdown("---")
    st.subheader("🔬 Model Performance Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **ResNet50 Performance:**
        - Test Accuracy: 92.63%
        - Benefits from ImageNet pretraining
        - Better for high-accuracy needs
        """)
    with col2:
        st.markdown("""
        **Custom CNN Performance:**
        - Test Accuracy: 88.21%
        - Lightweight and faster
        - Ideal for limited-resource environments
        """)

    with st.expander("📋 Technical Details"):
        st.markdown("""
        - Images resized to 224x224 and normalized
        - Models trained on brain MRI scans
        - 4 output classes: Glioma, Meningioma, No Tumor, Pituitary
        - Uses PyTorch with Streamlit frontend
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🧠 Brain Tumor Classification App | Built with Streamlit & PyTorch</p>
        <p><small>For educational purposes only - Not for medical diagnosis</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
