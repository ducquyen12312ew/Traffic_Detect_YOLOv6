import cv2
import numpy as np
from typing import List, Tuple
import os

def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [
        'data/images/train',
        'data/images/val', 
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/raw_videos',
        'data/output_videos',
        'models/pretrained',
        'models/custom',
        'models/weights'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def draw_bounding_box(image: np.ndarray, 
                     bbox: Tuple[int, int, int, int],
                     class_name: str,
                     confidence: float,
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Vẽ bounding box lên ảnh"""
    x1, y1, x2, y2 = bbox
    
    # Vẽ rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Vẽ label
    label = f"{class_name}: {confidence:.2f}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Background cho text
    cv2.rectangle(image, 
                 (x1, y1 - label_size[1] - 10),
                 (x1 + label_size[0], y1),
                 color, -1)
    
    # Text
    cv2.putText(image, label,
               (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 2)
    
    return image

def extract_frames_from_video(video_path: str, output_dir: str, frame_interval: int = 30):
    """Trích xuất frame từ video để tạo dataset"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

def resize_image_keep_ratio(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Resize ảnh giữ nguyên tỷ lệ"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Padding để đạt target_size
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    return padded