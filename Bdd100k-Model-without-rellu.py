import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import resnet18
import numpy as np
import shap
import cv2
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# --------------------------
# Step 1: Configuration
# --------------------------
class Config:
    IMAGE_DIR = "./BDDD100K/train/images"
    LABEL_FILE = "./BDDD100K/train/annotations/bdd100k_labels_images_train.json"
    SEG_LABEL_DIR = "bdd100k/labels/segmentation"
    NUM_CLASSES = 9  # [brake, steer_left, steer_right, accelerate, lane_change_left, lane_change_right, maintain_lane, stop_completely, overtake]
    INPUT_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_cuda_info(cls):
        print("\nCUDA Information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

config = Config()

# --------------------------
# Step 3: Model Definition
# --------------------------
def modify_relu_inplace(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model

def build_models():
    # Object Detection Model (YOLOv5)
    obj_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Segmentation Model (U-Net)
    seg_model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)

    # Decision Model
    decision_model = resnet18(pretrained=True)
    decision_model.fc = nn.Linear(decision_model.fc.in_features, config.NUM_CLASSES)

    # Disable in-place operations
    decision_model = modify_relu_inplace(decision_model)

    return obj_detector, seg_model.to(config.DEVICE), decision_model.to(config.DEVICE)

# --------------------------
# Step 5: Explainability with SHAP
# --------------------------
def explain_with_shap(decision_model, dataset):
    print("\nGenerating SHAP explanations...")
    decision_model.eval()

    print("Preparing background samples...")
    background = torch.stack([dataset[i][0] for i in range(100)]).to(config.DEVICE)

    print("Creating SHAP explainer...")
    explainer = shap.DeepExplainer(decision_model, background)

    print("Generating explanations for test images...")
    test_images = torch.stack([dataset[i][0] for i in range(5)]).to(config.DEVICE)

    # Clone test images to avoid in-place modification issues
    test_images = test_images.clone()
    shap_values = explainer.shap_values(test_images)

    print("Visualizing SHAP values...")
    for i in range(5):
        plt.figure(figsize=(5, 5))
        shap.image_plot(shap_values, np.transpose(test_images.cpu().numpy(), (0, 2, 3, 1)))
        plt.show()

# --------------------------
# Step 6: Main Pipeline
# --------------------------
if __name__ == "__main__":
    print("\n=== Starting BDD100K HMI Model Pipeline ===\n")

    # Add this right at the start
    Config.print_cuda_info()

    # Force CUDA memory clearance if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("1. Initializing Datasets...")
    train_dataset = BDD100KHMI(split='train')
    print(f"   - Train dataset size: {len(train_dataset)} samples")
    val_dataset = BDD100KHMI(split='val')
    print(f"   - Validation dataset size: {len(val_dataset)} samples")
    print("✓ Datasets initialized successfully\n")

    print("2. Creating DataLoaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print(f"   - Train batches: {len(train_dataloader)}")
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"   - Validation batches: {len(val_dataloader)}")
    print("✓ DataLoaders created successfully\n")

    print("3. Building Models...")
    print("   - Loading YOLOv5...")
    obj_detector, seg_model, decision_model = build_models()
    print("   - Loading Segmentation Model...")
    print("   - Loading Decision Model...")
    print(f"✓ All models loaded successfully (using {config.DEVICE})\n")

    print("4. Starting Model Training...")
    train_model(decision_model, train_dataloader)
    print("✓ Training completed\n")

    print("5. Generating SHAP Explanations...")
    explain_with_shap(decision_model, train_dataset)
    print("✓ SHAP analysis completed\n")

    print("6. Collecting Decision Labels...")
    decision_labels = train_dataset.get_all_labels()
    print("   Decision Labels Distribution:")
    unique_labels, counts = np.unique(decision_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   - Class {label}: {count} samples")
    print("✓ Label collection completed\n")

    print("=== Pipeline Completed Successfully ===")
