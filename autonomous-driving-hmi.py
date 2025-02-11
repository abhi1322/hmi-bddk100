# Import required libraries
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

class Config:
    """Configuration class containing all hyperparameters and settings"""
    def __init__(self):
        # Data paths
        self.IMAGE_DIR = "./BDDD100K/train/images"
        self.LABEL_FILE = "./BDDD100K/train/annotations/bdd100k_labels_images_train.json"
        self.SEG_LABEL_DIR = "bdd100k/labels/segmentation"
        
        # Model parameters
        self.NUM_CLASSES = 9  # Number of driving decisions
        self.INPUT_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.EPOCHS = 1
        self.LEARNING_RATE = 0.001
        
        # Device configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image normalization parameters
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]
        
        # Action mapping
        self.ACTION_MAP = {
            0: "brake",
            1: "steer_left",
            2: "steer_right",
            3: "accelerate",
            4: "lane_change_left",
            5: "lane_change_right",
            6: "maintain_lane",
            7: "stop_completely",
            8: "overtake"
        }
    
    def print_cuda_info(self):
        """Print CUDA device information"""
        print("\nCUDA Information:")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name()}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        else:
            print("CUDA is not available. Using CPU.")

class BDD100KHMI(Dataset):
    """Dataset class for BDD100K with HMI labels"""
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # Load annotations
        with open(config.LABEL_FILE, 'r') as f:
            self.data = json.load(f)
            
        # Define image transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(config.INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        entry = self.data[idx]
        
        # Load and transform image
        img_path = f"{self.config.IMAGE_DIR}/{entry['name']}"
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        
        # Determine driving action based on scene contents
        label = self._determine_action(entry)
        
        return image, label
    
    def _determine_action(self, entry):
        """Determine appropriate driving action based on scene contents"""
        for obj in entry.get('labels', []):
            # Priority-based decision making
            if obj['category'] == 'pedestrian':
                return 0  # Brake
            elif obj['category'] == 'stop_sign':
                return 7  # Stop completely
            elif obj['category'] == 'traffic light' and obj.get('attributes', {}).get('trafficLightColor') == 'red':
                return 0  # Brake
            # Add more conditions as needed
            
        return 6  # Default: maintain lane

class DrivingDecisionModel(nn.Module):
    """Neural network model for driving decisions"""
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained ResNet18 and modify for our use
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Disable inplace operations for SHAP compatibility
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
    
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc="Training"):
            images = images.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return running_loss / len(dataloader), 100 * correct / total

def explain_model_decisions(model, dataset, config):
    """Generate and visualize SHAP explanations"""
    model.eval()
    
    try:
        # Prepare data for SHAP
        background = torch.stack([dataset[i][0] for i in range(10)]).to(config.DEVICE)
        test_images = torch.stack([dataset[i][0] for i in range(3)]).to(config.DEVICE)
        
        # Create explainer and compute SHAP values
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_images)
        
        # Visualize results
        for idx in range(len(test_images)):
            visualize_explanation(
                original_image=test_images[idx],
                shap_values=shap_values,
                idx=idx,
                config=config
            )
            
    except Exception as e:
        print(f"Error in SHAP explanation: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_explanation(original_image, shap_values, idx, config):
    """Create visualization of SHAP explanations"""
    plt.figure(figsize=(15, 5))
    
    # Denormalize image for visualization
    img_display = denormalize_image(original_image, config)
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot SHAP values
    shap_magnitude = np.abs(np.array(shap_values)).sum(axis=0)[idx].sum(axis=2)
    plt.subplot(1, 3, 2)
    plt.imshow(shap_magnitude, cmap='hot')
    plt.title("SHAP Importance")
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_display)
    plt.imshow(shap_magnitude, cmap='hot', alpha=0.6)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def denormalize_image(image, config):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(config.NORMALIZE_STD).view(3, 1, 1)
    img = image.cpu().detach() * std + mean
    return img.permute(1, 2, 0).numpy()

def main():
    """Main execution function"""
    print("=== Starting BDD100K HMI Model Pipeline ===")
    
    # Initialize configuration
    config = Config()
    config.print_cuda_info()
    
    # Create datasets and dataloaders
    print("\nInitializing datasets...")
    train_dataset = BDD100KHMI(config, split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DrivingDecisionModel(config.NUM_CLASSES).to(config.DEVICE)
    trainer = ModelTrainer(model, config)
    
    # Train model
    print("\nStarting training...")
    for epoch in range(config.EPOCHS):
        loss, accuracy = trainer.train_epoch(train_dataloader)
        print(f"Epoch {epoch+1}/{config.EPOCHS}:")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Generate explanations
    print("\nGenerating model explanations...")
    explain_model_decisions(model, train_dataset, config)
    
    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
