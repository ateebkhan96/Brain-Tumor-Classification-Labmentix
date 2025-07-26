# 🧠 Brain Tumor Classification using Deep Learning

A comprehensive deep learning project for classifying brain MRI images into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**. This project includes both model training and an interactive Streamlit web application for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)

## 🎯 Overview

This project develops and compares two deep learning approaches for automated brain tumor classification:

1. **Transfer Learning with ResNet50**: Pre-trained model fine-tuned for medical imaging
2. **Custom CNN Architecture**: Built-from-scratch convolutional neural network

The models assist medical professionals by providing automated classification of brain MRI scans, potentially speeding up diagnosis and reducing human error.

## 📊 Dataset

The project uses a brain tumor MRI dataset with four classes:

| Class | Description | Training Samples |
|-------|-------------|------------------|
| **Glioma** | Most common primary brain tumor | ~1,321 |
| **Meningioma** | Tumor arising from brain coverings | ~1,339 |
| **No Tumor** | Normal brain tissue | ~1,595 |
| **Pituitary** | Pituitary gland tumor | ~1,457 |

### Data Preprocessing
- Images resized to 256×256 pixels
- Center cropped to 224×224 pixels
- Normalized using ImageNet statistics
- Data augmentation applied (rotation, flip, color jitter)

## 🤖 Models

### 1. ResNet50 (Transfer Learning)
- **Architecture**: Pre-trained ResNet50 with modified classifier
- **Approach**: Fine-tuning with frozen feature extraction layers
- **Final Layer**: Custom classifier with dropout regularization
- **Test Accuracy**: **92.63%**

### 2. Custom CNN
- **Architecture**: 4-layer convolutional neural network
- **Features**: Batch normalization, progressive channel increase (32→64→128→256)
- **Regularization**: Dropout layers to prevent overfitting
- **Test Accuracy**: **88.21%**

## ✨ Features

### Training Pipeline
- **Data Augmentation**: Rotation, flipping, color jittering
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Automatic saving of best models
- **Comprehensive Visualization**: Training curves and model comparisons

### Web Application
- **Interactive Interface**: User-friendly Streamlit web app
- **Model Selection**: Choose between ResNet50 and Custom CNN
- **Real-time Predictions**: Instant classification with confidence scores
- **Visual Results**: Progress bars showing probabilities for all classes
- **Professional UI**: Modern, responsive design with medical theming

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (recommended)
```

### Clone Repository
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
torch>=1.9.0
torchvision>=0.10.0
streamlit>=1.0.0
pillow>=8.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.60.0
```

## 📖 Usage

### 1. Training Models

#### Jupyter Notebook Training
```bash
# Open the training notebook
jupyter notebook "Brain Tumor deep seek.ipynb"

# Or use Google Colab for GPU access
# Upload the notebook to Google Colab and run all cells
```

#### Key Training Steps
1. **Data Loading**: Upload and extract dataset
2. **Exploration**: Analyze class distribution and sample images
3. **Preprocessing**: Apply transformations and create data loaders
4. **Model Training**: Train both ResNet50 and Custom CNN
5. **Evaluation**: Generate classification reports and confusion matrices
6. **Model Saving**: Save trained models as `.pth` files

### 2. Model Saving
The training notebook automatically saves models in the `saved_models/` directory:

```python
# Models are saved with this structure:
{
    'model_state_dict': model.state_dict(),
    'class_names': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
    'input_size': 224
}
```

**Saved Files:**
- `saved_models/resnet50_brain_tumor.pth`
- `saved_models/custom_cnn_brain_tumor.pth`

## 🌐 Web Application

### Launch Streamlit App
```bash
streamlit run app.py
```

### Using the Web App

1. **Select Model**: Choose between ResNet50 or Custom CNN
2. **Upload Model**: Upload the corresponding `.pth` file
3. **Upload Image**: Select a brain MRI scan (JPG, PNG, JPEG)
4. **Get Results**: View classification with confidence scores

### App Features
- ✅ **Model Comparison**: Switch between different architectures
- ✅ **Confidence Visualization**: Progress bars for all class probabilities
- ✅ **Medical Interface**: Professional, clinical-appropriate design
- ✅ **Error Handling**: Robust error messages and validation
- ✅ **Mobile Responsive**: Works on desktop, tablet, and mobile



## 📁 Project Structure

```
brain-tumor-classification/
│
├── Brain Tumor deep seek.ipynb    # Main training notebook
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
│
├── saved_models/                 # Trained model files
│   ├── resnet50_brain_tumor.pth
│   └── custom_cnn_brain_tumor.pth
│
├── screenshots/                  # App screenshots
│   ├── main_interface.png
│   └── prediction_results.png
│
└── sample_images/               # Sample MRI images for testing
    ├── glioma_sample.jpg
    ├── meningioma_sample.jpg
    ├── no_tumor_sample.jpg
    └── pituitary_sample.jpg
```

## 🔧 Technical Details

### Model Architectures

#### ResNet50 Architecture
```python
ResNet50(
  (conv1): Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
  (bn1): BatchNorm2d(64)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  # ... ResNet blocks ...
  (fc): Sequential(
    (0): Linear(2048, 512)
    (1): ReLU()
    (2): Dropout(0.5)
    (3): Linear(512, 4)
  )
)
```

#### Custom CNN Architecture
```python
BrainTumorCNN(
  (features): Sequential(
    # Block 1: 3→32 channels
    (0): Conv2d(3, 32, kernel_size=3, padding=1)
    (1): BatchNorm2d(32)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2)
    # ... additional blocks ...
  )
  (classifier): Sequential(
    (0): Linear(50176, 512)
    (1): ReLU()
    (2): Dropout(0.5)
    (3): Linear(512, 4)
  )
)
```

### Training Configuration
- **Optimizer**: Adam with different learning rates
  - ResNet50: 0.0001 (fine-tuning)
  - Custom CNN: 0.001 (training from scratch)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Early Stopping**: Patience of 5 epochs
- **Learning Rate Scheduler**: ReduceLROnPlateau

### Data Augmentation Strategy
```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
