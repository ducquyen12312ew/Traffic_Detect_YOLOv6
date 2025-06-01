import cv2
import numpy as np
from typing import List, Tuple, Dict  # Added Dict import
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
    """Vẽ bounding box lên ảnh với Vietnamese text support"""
    x1, y1, x2, y2 = bbox
    
    # Vẽ rectangle với border dày hơn
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    # Format label - Vietnamese friendly
    if confidence < 1.0:
        label = f"{class_name}: {confidence:.2f}"
    else:
        label = f"{class_name}: 1.00"
    
    # Calculate label size với font size lớn hơn
    font_scale = 0.7
    font_thickness = 2
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    
    # Ensure label không bị cắt
    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
    
    # Background cho text với padding
    padding = 5
    cv2.rectangle(image, 
                 (x1, label_y - label_size[1] - padding),
                 (x1 + label_size[0] + padding, label_y + padding),
                 color, -1)
    
    # Border cho text background
    cv2.rectangle(image, 
                 (x1, label_y - label_size[1] - padding),
                 (x1 + label_size[0] + padding, label_y + padding),
                 (255, 255, 255), 1)
    
    # Text màu trắng với font dày
    cv2.putText(image, label,
               (x1 + 2, label_y - 2),
               cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, (255, 255, 255), font_thickness)
    
    return image

def draw_traffic_light_info(image: np.ndarray, traffic_status: Dict[str, int], 
                          position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Vẽ thông tin traffic light status lên ảnh
    """
    x, y = position
    
    # Background cho info panel
    panel_width = 250
    panel_height = len([k for k, v in traffic_status.items() if v > 0]) * 30 + 20
    
    if panel_height > 20:  # Chỉ vẽ nếu có traffic lights
        cv2.rectangle(image, (x-5, y-15), (x + panel_width, y + panel_height), 
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(image, (x-5, y-15), (x + panel_width, y + panel_height), 
                     (255, 255, 255), 2)  # White border
        
        # Traffic light status
        offset = 0
        for color, count in traffic_status.items():
            if count > 0:
                # Vietnamese names
                vietnamese_names = {
                    'red': 'Đèn Đỏ',
                    'yellow': 'Đèn Vàng', 
                    'green': 'Đèn Xanh',
                    'unknown': 'Đèn Không Rõ'
                }
                
                display_name = vietnamese_names.get(color, color)
                text = f"{display_name}: {count}"
                
                # Color coding
                if color == 'red':
                    text_color = (0, 0, 255)
                elif color == 'yellow':
                    text_color = (0, 255, 255)
                elif color == 'green':
                    text_color = (0, 255, 0)
                else:
                    text_color = (255, 255, 255)
                
                cv2.putText(image, text, (x, y + offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                offset += 25
    
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