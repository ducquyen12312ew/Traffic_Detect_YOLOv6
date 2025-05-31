#!/usr/bin/env python3
"""
train_vietnam_signs.py - Script để train CNN model cho biển báo giao thông Việt Nam
Sử dụng PyTorch, tương thích với Python 3.13.1
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import glob
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt

class VietnamSignDataset(Dataset):
    """Dataset class cho biển báo Việt Nam"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image for transforms
        if self.transform:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Convert single channel to 3 channel for transforms
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            image = self.transform(image)
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        
        return image, torch.tensor(label, dtype=torch.long)

class VietnamSignCNN(nn.Module):
    """
    CNN model cho biển báo Việt Nam - Architecture giống document TensorFlow
    """
    def __init__(self, num_classes=8):
        super(VietnamSignCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Conv Block 3 & 4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # 64/8 = 8 after 3 pooling layers
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool3(x)
        
        # Dense layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        
        return x

class VietnamSignTrainer:
    def __init__(self, data_dir: str = "data/vietnam_signs"):
        """
        Khởi tạo trainer cho biển báo Việt Nam
        """
        self.data_dir = data_dir
        self.input_shape = (1, 64, 64)  # PyTorch format: (C, H, W)
        self.num_classes = 8
        self.batch_size = 128
        self.epochs = 15
        self.learning_rate = 0.001
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Folder mapping từ document (folder_name = [0,2,6,10,14,22,33,34])
        self.folder_mapping = {
            0: 0,   # Class 0 - Biển báo không xác định
            2: 1,   # Class 1 - Cấm đi ngược chiều
            6: 2,   # Class 2 - Giới hạn tốc độ
            10: 3,  # Class 3 - Cấm rẽ
            14: 4,  # Class 4 - Biển cảnh báo
            22: 5,  # Class 5 - Biển chỉ dẫn
            33: 6,  # Class 6 - Biển báo thông tin
            34: 7   # Class 7 - Biển phụ
        }
        
        self.class_names = [
            "unknown_sign",      # 0
            "no_entry",          # 2
            "speed_limit",       # 6
            "no_turn",           # 10
            "warning_sign",      # 14
            "mandatory_sign",    # 22
            "information_sign",  # 33
            "additional_sign"    # 34
        ]
        
        self.vietnam_names = [
            "Biển báo không xác định",
            "Cấm đi ngược chiều",
            "Giới hạn tốc độ", 
            "Cấm rẽ",
            "Biển cảnh báo",
            "Biển chỉ dẫn",
            "Biển báo thông tin",
            "Biển phụ"
        ]
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Tiền xử lý ảnh theo format document (giống TensorFlow version)
        """
        # Đọc ảnh grayscale
        img = cv2.imread(image_path, 0)
        if img is None:
            return None
        
        # Resize về 64x64
        img = cv2.resize(img, (64, 64))
        
        # Histogram equalization (giống document)
        img = cv2.equalizeHist(img)
        
        # Normalize về [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_data_from_folders(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load dữ liệu từ folder structure
        Expected structure:
        data/vietnam_signs/
        ├── 0/    (Biển báo không xác định)
        ├── 2/    (Cấm đi ngược chiều)
        ├── 6/    (Giới hạn tốc độ)
        ├── 10/   (Cấm rẽ)
        ├── 14/   (Biển cảnh báo)
        ├── 22/   (Biển chỉ dẫn)
        ├── 33/   (Biển báo thông tin)
        └── 34/   (Biển phụ)
        """
        X_train = []
        y_train = []
        y_labels = []
        
        # Folders từ document
        folder_names = [0, 2, 6, 10, 14, 22, 33, 34]
        
        for i, folder_name in enumerate(folder_names):
            folder_path = os.path.join(self.data_dir, str(folder_name))
            
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found")
                continue
            
            # Tìm tất cả file ảnh
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
                image_files.extend(glob.glob(os.path.join(folder_path, pattern.upper())))
            
            print(f"Loading {len(image_files)} images from folder {folder_name} ({self.vietnam_names[i]})")
            
            tmp_length = 0
            for image_file in image_files:
                processed_img = self.preprocess_image(image_file)
                if processed_img is not None:
                    X_train.append(processed_img)
                    tmp_length += 1
            
            # Add labels
            y_train.extend([i] * tmp_length)
            y_labels.append(folder_name)
            
            print(f"Loaded {tmp_length} images for class {i}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"\nTotal training samples: {len(X_train)}")
        print(f"Data shape: {X_train.shape}")
        print(f"Labels shape: {y_train.shape}")
        
        return X_train, y_train, y_labels
    
    def train_model(self, save_path: str = "models/vietnam_signs_model.pth") -> nn.Module:
        """
        Train model CNN với PyTorch
        """
        print("=" * 60)
        print("VIETNAM TRAFFIC SIGNS TRAINING")
        print("=" * 60)
        
        print("Loading training data...")
        X_train, y_train, y_labels = self.load_data_from_folders()
        
        if len(X_train) == 0:
            raise ValueError("No training data found!")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=1
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Create datasets
        train_dataset = VietnamSignDataset(X_train, y_train, self.transform)
        val_dataset = VietnamSignDataset(X_val, y_val, self.transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        model = VietnamSignCNN(num_classes=self.num_classes)
        model.to(self.device)
        
        # Loss function và optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.2)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {total_params:,}")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Store history
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f'\nEpoch {epoch+1}/{self.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f'  ✓ New best model saved: {val_acc:.2f}%')
            
            print('-' * 60)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {save_path}")
        
        # Plot training history
        self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        return model
    
    def plot_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """
        Vẽ biểu đồ training history
        """
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy
        ax1.plot(train_accs, label='Training Accuracy', linewidth=2, color='blue')
        ax1.plot(val_accs, label='Validation Accuracy', linewidth=2, color='red')
        ax1.set_title('Model Accuracy - Vietnam Traffic Signs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Loss
        ax2.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
        ax2.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        ax2.set_title('Model Loss - Vietnam Traffic Signs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vietnam_signs_training_history.png', dpi=300, bbox_inches='tight')
        print("📊 Training history saved to: vietnam_signs_training_history.png")
        plt.show()
    
    def create_sample_data_structure(self):
        """
        Tạo cấu trúc thư mục mẫu cho training data
        """
        base_dir = self.data_dir
        folder_names = [0, 2, 6, 10, 14, 22, 33, 34]
        
        print("Creating sample data structure...")
        
        for i, folder_name in enumerate(folder_names):
            folder_path = os.path.join(base_dir, str(folder_name))
            os.makedirs(folder_path, exist_ok=True)
            
            # Tạo file README
            readme_path = os.path.join(folder_path, "README.txt")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"FOLDER {folder_name} - CLASS {i}\n")
                f.write("=" * 40 + "\n")
                f.write(f"English: {self.class_names[i]}\n")
                f.write(f"Vietnamese: {self.vietnam_names[i]}\n")
                f.write("\nInstructions:\n")
                f.write("- Put your training images here\n")
                f.write("- Supported formats: jpg, jpeg, png, bmp\n")
                f.write("- Images should contain traffic signs only\n")
                f.write("- Recommended: at least 100 images per class\n")
        
        print(f"📁 Data structure created in: {base_dir}")
        print("\n📋 Folder structure:")
        for i, folder_name in enumerate(folder_names):
            print(f"  {folder_name}/ - {self.vietnam_names[i]}")
        
        print("\n📝 Next steps:")
        print("1. Add training images to respective folders")
        print("2. Run: python train_vietnam_signs.py")

def main():
    """
    Main function để chạy training
    """
    print("🇻🇳 VIETNAM TRAFFIC SIGNS TRAINER")
    print("Using PyTorch - Compatible with Python 3.13.1")
    print("=" * 60)
    
    # Tạo trainer
    trainer = VietnamSignTrainer()
    
    # Tạo data structure nếu chưa có
    if not os.path.exists(trainer.data_dir):
        trainer.create_sample_data_structure()
        print("\n⚠️  Please add training images and run again.")
        return
    
    # Kiểm tra có data không
    folder_names = [0, 2, 6, 10, 14, 22, 33, 34]
    total_images = 0
    
    print("📊 Checking data availability:")
    for i, folder_name in enumerate(folder_names):
        folder_path = os.path.join(trainer.data_dir, str(folder_name))
        if os.path.exists(folder_path):
            image_files = []
            for pattern in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
                image_files.extend(glob.glob(os.path.join(folder_path, pattern.upper())))
            print(f"  Folder {folder_name}: {len(image_files)} images ({trainer.vietnam_names[i]})")
            total_images += len(image_files)
        else:
            print(f"  Folder {folder_name}: Not found")
    
    if total_images == 0:
        print("\n❌ No training images found!")
        print("Please add images to the folders and run again.")
        return
    
    print(f"\n✅ Found {total_images} total training images")
    
    # Confirm training
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    try:
        # Train model
        model = trainer.train_model()
        
        # Test inference
        print("\n🧪 Testing model inference...")
        test_input = torch.randn(1, 1, 64, 64).to(trainer.device)
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            predicted = torch.argmax(output, dim=1)
            probabilities = torch.softmax(output, dim=1)
            print(f"Test prediction: Class {predicted.item()}")
            print(f"Confidence: {probabilities[0][predicted].item():.4f}")
        
        print("\n🎉 Training completed successfully!")
        print("Model ready for inference!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()