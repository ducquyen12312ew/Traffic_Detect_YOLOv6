# File src/train_model.py
from ultralytics import YOLO

def train_custom_model():
    model = YOLO('yolov8n.pt')  # Load pretrained model
    
    # Train
    results = model.train(
        data='configs/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='traffic_detection'
    )
    
    return results

if __name__ == "__main__":
    train_custom_model()