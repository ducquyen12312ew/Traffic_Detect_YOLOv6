import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import os
import torch
import glob
from datetime import datetime
from .config import Config
from .utils import draw_bounding_box
from .traffic_light_detector import EnhancedTrafficLightDetector

class TrafficDetector:
    def __init__(self, model_path: str = None, sign_model_path: str = None):
        """
        Khởi tạo Traffic Detector - Simplified version cho Python 3.13.1
        Args:
            model_path: Đường dẫn đến model custom YOLO
            sign_model_path: Tạm thời không sử dụng (TensorFlow incompatible)
        """
        # Force CPU if GPU causes issues
        device = 'cpu'  # Hoặc 'cuda' nếu muốn dùng GPU
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model from {model_path}")
        else:
            self.model = YOLO(f'{Config.MODEL_SIZE}.pt')
            print(f"Loaded pretrained model: {Config.MODEL_SIZE}")
        
        # Set device
        self.model.to(device)
        print(f"Using device: {device}")
            
        self.config = Config()
        
        # Initialize traffic light color detector
        self.traffic_light_detector = EnhancedTrafficLightDetector()
        
        # Vietnamese sign classifier tạm thời disabled
        self.sign_detector = None
        if sign_model_path:
            print(f"Vietnamese sign classification disabled (TensorFlow not compatible with Python 3.13)")
            print(f"Please use Python 3.11 for full functionality")
        
        # Simple tracking for video stability
        self.tracked_objects = {}
        self.next_id = 0
        self.max_disappeared = 5  # Reduced for faster response
        self.max_distance = 80  # Reduced for tighter tracking
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh trước khi detect
        """
        # Kiểm tra input
        if image is None:
            return None
            
        # Resize nếu ảnh quá lớn
        height, width = image.shape[:2]
        max_size = 1280
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Đảm bảo image có 3 channels
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image
        elif len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            return None
    
    def get_basic_sign_classification(self, class_name: str, roi: np.ndarray = None) -> Dict:
        """
        Enhanced basic classification cho traffic signs
        """
        # Enhanced sign keywords với more details
        sign_keywords = {
            'stop': {'vietnamese_name': 'Biển báo dừng (STOP)', 'color': (0, 0, 255), 'confidence_boost': 0.2},
            'speed': {'vietnamese_name': 'Biển giới hạn tốc độ', 'color': (255, 0, 0), 'confidence_boost': 0.15},
            'yield': {'vietnamese_name': 'Biển nhường đường', 'color': (0, 255, 255), 'confidence_boost': 0.15},
            'no_entry': {'vietnamese_name': 'Biển cấm đi ngược chiều', 'color': (0, 0, 255), 'confidence_boost': 0.15},
            'warning': {'vietnamese_name': 'Biển cảnh báo', 'color': (0, 255, 255), 'confidence_boost': 0.1},
            'mandatory': {'vietnamese_name': 'Biển chỉ dẫn bắt buộc', 'color': (255, 0, 0), 'confidence_boost': 0.1},
            'information': {'vietnamese_name': 'Biển thông tin hướng dẫn', 'color': (0, 255, 0), 'confidence_boost': 0.1},
            'pedestrian': {'vietnamese_name': 'Biển báo người đi bộ', 'color': (255, 255, 0), 'confidence_boost': 0.1},
            'school': {'vietnamese_name': 'Khu vực trường học', 'color': (0, 255, 0), 'confidence_boost': 0.15},
            'construction': {'vietnamese_name': 'Khu vực thi công', 'color': (0, 255, 255), 'confidence_boost': 0.1},
            'parking': {'vietnamese_name': 'Biển báo đỗ xe', 'color': (255, 0, 255), 'confidence_boost': 0.1},
            'turn': {'vietnamese_name': 'Biển báo rẽ', 'color': (128, 255, 0), 'confidence_boost': 0.1}
        }
        
        # Enhanced analysis nếu có ROI
        if roi is not None and roi.size > 0:
            # Analyze colors in ROI để improve classification
            roi_analysis = self.analyze_sign_roi(roi)
        else:
            roi_analysis = {'dominant_color': 'unknown', 'shape_hint': 'unknown'}
        
        # Find best match
        class_lower = class_name.lower()
        best_match = None
        best_score = 0
        
        for keyword, info in sign_keywords.items():
            if keyword in class_lower:
                score = len(keyword) / len(class_lower)  # Longer matches get higher score
                if score > best_score:
                    best_score = score
                    best_match = info
        
        if best_match:
            result = best_match.copy()
            
            # Enhance based on ROI analysis
            if roi_analysis['dominant_color'] == 'red':
                if 'cấm' not in result['vietnamese_name'].lower():
                    result['vietnamese_name'] = f"Biển cấm - {result['vietnamese_name']}"
                result['confidence_boost'] += 0.05
            elif roi_analysis['dominant_color'] == 'yellow':
                if 'cảnh báo' not in result['vietnamese_name'].lower():
                    result['vietnamese_name'] = f"Biển cảnh báo - {result['vietnamese_name']}"
                result['confidence_boost'] += 0.05
            
            return result
        
        # Default classification
        return {
            'vietnamese_name': 'Biển báo giao thông',
            'color': (128, 128, 128),
            'confidence_boost': 0.0
        }
    
    def analyze_sign_roi(self, roi: np.ndarray) -> Dict:
        """
        Analyze sign ROI để get color và shape hints
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Color analysis
            color_ranges = {
                'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                       (np.array([170, 100, 100]), np.array([180, 255, 255]))],
                'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
                'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
                'green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
                'white': [(np.array([0, 0, 200]), np.array([180, 30, 255]))]
            }
            
            color_scores = {}
            total_pixels = roi.shape[0] * roi.shape[1]
            
            for color_name, ranges in color_ranges.items():
                total_area = 0
                for lower, upper in ranges:
                    mask = cv2.inRange(hsv, lower, upper)
                    total_area += cv2.countNonZero(mask)
                color_scores[color_name] = total_area / total_pixels
            
            dominant_color = max(color_scores, key=color_scores.get) if color_scores else 'unknown'
            
            return {
                'dominant_color': dominant_color,
                'color_percentages': color_scores,
                'shape_hint': 'unknown'  # Could add shape analysis here
            }
            
        except Exception as e:
            return {'dominant_color': 'unknown', 'shape_hint': 'unknown'}
        
    def track_objects(self, detections: List[Dict]) -> List[Dict]:
        """
        Advanced object tracking với size adaptation và smooth movement
        """
        if not detections:
            # Mark all objects as disappeared
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['disappeared'] += 1
                if self.tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
            return []
        
        # Calculate centers và areas for detections
        detection_info = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            detection_info.append({
                'center': (center_x, center_y),
                'size': (width, height),
                'area': area,
                'detection': detection
            })
        
        # If no tracked objects, register all as new
        if not self.tracked_objects:
            tracked_detections = []
            for i, info in enumerate(detection_info):
                detection = info['detection']
                self.tracked_objects[self.next_id] = {
                    'bbox': detection['bbox'],
                    'center': info['center'],
                    'size': info['size'],
                    'area': info['area'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'stable_frames': 0,
                    'velocity': (0, 0),  # Track movement
                    'size_history': [info['area']]  # Track size changes
                }
                
                tracked_detection = detection.copy()
                tracked_detection['track_id'] = self.next_id
                tracked_detection['stable'] = False
                tracked_detections.append(tracked_detection)
                
                self.next_id += 1
            
            return tracked_detections
        
        # Advanced matching với size và class compatibility
        tracked_detections = []
        used_detections = set()
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            best_match_idx = -1
            best_score = float('inf')
            
            # Find best matching detection
            for i, info in enumerate(detection_info):
                if i in used_detections:
                    continue
                
                detection = info['detection']
                
                # Skip if class doesn't match (với tolerance)
                if (tracked_obj['class_name'] != detection['class_name'] and 
                    tracked_obj['stable_frames'] > 5):
                    continue
                
                # Calculate distance score
                distance = np.sqrt((info['center'][0] - tracked_obj['center'][0])**2 + 
                                 (info['center'][1] - tracked_obj['center'][1])**2)
                
                # Calculate size compatibility score
                size_ratio = min(info['area'] / max(tracked_obj['area'], 1), 
                               max(tracked_obj['area'], 1) / info['area'])
                
                # Predict position based on velocity
                predicted_x = tracked_obj['center'][0] + tracked_obj['velocity'][0]
                predicted_y = tracked_obj['center'][1] + tracked_obj['velocity'][1]
                predicted_distance = np.sqrt((info['center'][0] - predicted_x)**2 + 
                                           (info['center'][1] - predicted_y)**2)
                
                # Combined score (lower is better)
                score = (distance * 0.7 + predicted_distance * 0.3) / max(size_ratio, 0.1)
                
                if score < self.max_distance and score < best_score:
                    best_score = score
                    best_match_idx = i
            
            if best_match_idx != -1:
                # Update tracked object với advanced smoothing
                info = detection_info[best_match_idx]
                detection = info['detection']
                
                # Calculate velocity
                old_center = tracked_obj['center']
                new_center = info['center']
                velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                
                # Smooth velocity
                old_velocity = tracked_obj['velocity']
                smoothed_velocity = (
                    int(0.3 * velocity[0] + 0.7 * old_velocity[0]),
                    int(0.3 * velocity[1] + 0.7 * old_velocity[1])
                )
                
                # Smooth position với prediction
                alpha = 0.8 if tracked_obj['stable_frames'] > 3 else 0.6
                smoothed_center = (
                    int(alpha * new_center[0] + (1 - alpha) * old_center[0]),
                    int(alpha * new_center[1] + (1 - alpha) * old_center[1])
                )
                
                # Smooth size changes
                old_size = tracked_obj['size']
                new_size = info['size']
                
                # Size smoothing với adaptive rate
                size_alpha = 0.7 if abs(info['area'] - tracked_obj['area']) < tracked_obj['area'] * 0.3 else 0.4
                smoothed_size = (
                    int(size_alpha * new_size[0] + (1 - size_alpha) * old_size[0]),
                    int(size_alpha * new_size[1] + (1 - size_alpha) * old_size[1])
                )
                
                # Ensure minimum size
                smoothed_size = (max(smoothed_size[0], 20), max(smoothed_size[1], 20))
                
                # Create smoothed bbox
                smoothed_bbox = (
                    smoothed_center[0] - smoothed_size[0] // 2,
                    smoothed_center[1] - smoothed_size[1] // 2,
                    smoothed_center[0] + smoothed_size[0] // 2,
                    smoothed_center[1] + smoothed_size[1] // 2
                )
                
                # Update size history
                size_history = tracked_obj['size_history'][-5:] + [info['area']]  # Keep last 5
                
                # Update tracked object
                self.tracked_objects[obj_id].update({
                    'bbox': smoothed_bbox,
                    'center': smoothed_center,
                    'size': smoothed_size,
                    'area': smoothed_size[0] * smoothed_size[1],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'stable_frames': min(tracked_obj['stable_frames'] + 1, 20),
                    'velocity': smoothed_velocity,
                    'size_history': size_history
                })
                
                # Create tracked detection
                tracked_detection = detection.copy()
                tracked_detection['bbox'] = smoothed_bbox
                tracked_detection['track_id'] = obj_id
                tracked_detection['stable'] = tracked_obj['stable_frames'] >= 5
                tracked_detection['velocity'] = smoothed_velocity
                tracked_detection['size_stable'] = len(size_history) >= 3
                
                tracked_detections.append(tracked_detection)
                used_detections.add(best_match_idx)
            else:
                # No match found, predict position
                if tracked_obj['stable_frames'] > 3:
                    # Use velocity to predict position
                    predicted_center = (
                        tracked_obj['center'][0] + tracked_obj['velocity'][0],
                        tracked_obj['center'][1] + tracked_obj['velocity'][1]
                    )
                    
                    # Create predicted bbox
                    size = tracked_obj['size']
                    predicted_bbox = (
                        predicted_center[0] - size[0] // 2,
                        predicted_center[1] - size[1] // 2,
                        predicted_center[0] + size[0] // 2,
                        predicted_center[1] + size[1] // 2
                    )
                    
                    # Update with prediction
                    self.tracked_objects[obj_id].update({
                        'bbox': predicted_bbox,
                        'center': predicted_center,
                        'disappeared': tracked_obj['disappeared'] + 1
                    })
                    
                    # Add predicted detection if still stable
                    if tracked_obj['disappeared'] < 3:
                        predicted_detection = {
                            'bbox': predicted_bbox,
                            'class_name': tracked_obj['class_name'],
                            'confidence': max(tracked_obj['confidence'] - 0.1, 0.3),
                            'track_id': obj_id,
                            'stable': True,
                            'predicted': True
                        }
                        tracked_detections.append(predicted_detection)
                else:
                    # Mark as disappeared
                    self.tracked_objects[obj_id]['disappeared'] += 1
        
        # Register new detections với stricter criteria
        for i, info in enumerate(detection_info):
            if i not in used_detections:
                detection = info['detection']
                
                # Only register if confidence is high enough or area is significant
                if detection['confidence'] > 0.5 or info['area'] > 2000:
                    self.tracked_objects[self.next_id] = {
                        'bbox': detection['bbox'],
                        'center': info['center'],
                        'size': info['size'],
                        'area': info['area'],
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                        'disappeared': 0,
                        'stable_frames': 0,
                        'velocity': (0, 0),
                        'size_history': [info['area']]
                    }
                    
                    tracked_detection = detection.copy()
                    tracked_detection['track_id'] = self.next_id
                    tracked_detection['stable'] = False
                    tracked_detections.append(tracked_detection)
                    
                    self.next_id += 1
        
        # Remove disappeared objects
        for obj_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                del self.tracked_objects[obj_id]
        
        # Only return stable và high-confidence objects
        final_detections = []
        for d in tracked_detections:
            if (d.get('stable', False) or 
                d['confidence'] > 0.7 or 
                d.get('predicted', False)):
                final_detections.append(d)
        
        return final_detections

    def detect_image(self, image: np.ndarray, use_tracking: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects trong một ảnh - FIXED VERSION với better filtering
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return image, []
        
        detections = []
        annotated_image = image.copy()
        
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Detect với stricter parameters
            results = self.model.predict(
                processed_image,
                conf=0.5,  # Tăng threshold cao hơn
                iou=0.6,   # Tăng IOU để merge better
                verbose=False,
                save=False,
                imgsz=640,
                agnostic_nms=True,
                max_det=20  # Limit số detections
            )
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Define relevant classes for traffic (COCO dataset)
                relevant_classes = {
                    0: 'person',      # Người
                    1: 'bicycle',     # Xe đạp  
                    2: 'car',         # Xe hơi
                    3: 'motorcycle',  # Xe máy
                    5: 'bus',         # Xe buýt
                    7: 'truck',       # Xe tải
                    9: 'traffic_light' # Đèn giao thông
                }
                
                for i in range(len(boxes)):
                    try:
                        box = boxes[i]
                        
                        # Coordinates
                        coords = box.xyxy[0].cpu().numpy() if box.xyxy.is_cuda else box.xyxy[0].numpy()
                        x1, y1, x2, y2 = coords.astype(int)
                        
                        # Confidence
                        conf = box.conf[0].cpu().numpy() if box.conf.is_cuda else box.conf[0].numpy()
                        confidence = float(conf)
                        
                        # Class
                        cls = box.cls[0].cpu().numpy() if box.cls.is_cuda else box.cls[0].numpy()
                        class_id = int(cls)
                        
                        # ONLY process relevant traffic classes
                        if class_id not in relevant_classes:
                            continue
                        
                        class_name = relevant_classes[class_id]
                        
                        # Validate coordinates
                        h, w = image.shape[:2]
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))
                        
                        # Strict size filtering
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Minimum size requirements per class
                        min_sizes = {
                            'person': 30,
                            'bicycle': 40,
                            'car': 50,
                            'motorcycle': 35,
                            'bus': 60,
                            'truck': 60,
                            'traffic_light': 20
                        }
                        
                        min_size = min_sizes.get(class_name, 30)
                        if width < min_size or height < min_size:
                            continue
                        
                        # Aspect ratio filtering per class
                        aspect_ratio = width / height
                        
                        valid_ratios = {
                            'person': (0.3, 2.0),     # Người thường đứng
                            'bicycle': (0.7, 2.0),    # Xe đạp
                            'car': (1.2, 3.0),        # Xe hơi ngang
                            'motorcycle': (0.8, 2.5), # Xe máy
                            'bus': (2.0, 4.0),        # Xe buýt dài
                            'truck': (1.5, 4.0),      # Xe tải
                            'traffic_light': (0.3, 3.0) # Đèn có thể dọc/ngang
                        }
                        
                        min_ratio, max_ratio = valid_ratios.get(class_name, (0.2, 5.0))
                        if not (min_ratio <= aspect_ratio <= max_ratio):
                            continue
                        
                        # Position filtering - avoid objects at very top of image
                        if y1 < h * 0.1:  # Top 10% of image
                            continue
                        
                        # Size reasonableness check
                        obj_area = width * height
                        img_area = w * h
                        area_ratio = obj_area / img_area
                        
                        # Skip objects that are too large (likely background)
                        if area_ratio > 0.3:  # More than 30% of image
                            continue
                        
                        # Skip objects that are too small relative to image
                        if area_ratio < 0.0005:  # Less than 0.05% of image
                            continue
                        
                        # Calculate enhanced metrics
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Distance estimation based on object size and type
                        expected_sizes = {
                            'person': 20000,    # Expected pixel area for person at ~10m
                            'car': 40000,       # Expected pixel area for car at ~10m
                            'motorcycle': 15000,
                            'bicycle': 12000,
                            'bus': 80000,
                            'truck': 60000,
                            'traffic_light': 5000
                        }
                        
                        expected_size = expected_sizes.get(class_name, 20000)
                        distance_factor = max(0.1, min(1.0, expected_size / obj_area))
                        estimated_distance = int(10 * distance_factor)
                        
                        # Risk assessment - only for moving objects
                        if class_name in ['person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck']:
                            # Distance to image center (collision path)
                            img_center_x = w // 2
                            distance_to_center = abs(center_x - img_center_x) / (w // 2)
                            
                            # Vertical position (lower in image = closer)
                            vertical_position = y2 / h
                            
                            # Size factor (larger = closer = more dangerous)
                            size_factor = min(1.0, obj_area / (w * h * 0.1))
                            
                            # Calculate risk
                            risk_score = (1.0 - distance_to_center) * 0.4 + size_factor * 0.4 + vertical_position * 0.2
                            
                            if risk_score < 0.3:
                                risk_level = "low"
                            elif risk_score < 0.6:
                                risk_level = "medium"
                            else:
                                risk_level = "high"
                        else:
                            # Static objects (traffic lights) have no collision risk
                            risk_score = 0.0
                            risk_level = "low"
                        
                        # Store detection with clean data
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'class_name': class_name,
                            'confidence': confidence,
                            'class_id': class_id,
                            'estimated_distance': estimated_distance,
                            'risk_score': risk_score,
                            'risk_level': risk_level,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height,
                            'area': obj_area
                        }
                        detections.append(detection)
                        
                    except Exception as e:
                        print(f"Error processing detection {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in detection: {e}")
            return image, []
        
        # Apply enhanced NMS
        detections = self._apply_enhanced_nms(detections, 0.3)
        
        # Traffic light color analysis
        enhanced_detections = self.traffic_light_detector.analyze_traffic_lights(image, detections)
        
        # Apply tracking for video
        if use_tracking:
            enhanced_detections = self.track_objects(enhanced_detections)
        
        # Vietnamese display names
        vietnamese_names = {
            'person': 'Người',
            'car': 'Xe hơi',
            'truck': 'Xe tải',
            'bus': 'Xe buýt',
            'motorcycle': 'Xe máy',
            'bicycle': 'Xe đạp',
            'traffic_light': 'Đèn giao thông'
        }
        
        # Draw clean results
        annotated_image = image.copy()
        
        for detection in enhanced_detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            risk_level = detection.get('risk_level', 'low')
            
            # Color based on object type and risk
            if class_name == 'traffic_light':
                if 'traffic_light_color' in detection:
                    light_color = detection['traffic_light_color']
                    if light_color == 'red':
                        color = (0, 0, 255)
                    elif light_color == 'yellow':
                        color = (0, 255, 255)
                    elif light_color == 'green':
                        color = (0, 255, 0)
                    else:
                        color = (128, 128, 128)
                    display_name = f"Đèn {light_color.upper()}"
                else:
                    color = (128, 128, 128)
                    display_name = vietnamese_names.get(class_name, class_name)
            else:
                # Risk-based coloring for vehicles and people
                if risk_level == 'high':
                    color = (0, 0, 255)  # Red
                elif risk_level == 'medium':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                
                display_name = vietnamese_names.get(class_name, class_name)
            
            # Box thickness based on stability
            thickness = 3 if detection.get('stable', False) else 2
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Create clean label
            label = f"{display_name}: {confidence:.2f}"
            
            # Add distance for moving objects
            if class_name != 'traffic_light':
                label += f", ~{detection['estimated_distance']}m"
            
            # Add tracking info
            if detection.get('stable', False):
                label += " ✓"
            
            # Add risk warning for high risk objects
            if risk_level == 'high' and class_name != 'traffic_light':
                label += " ⚠️"
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1),
                         color, -1)
            
            # Text
            cv2.putText(annotated_image, label,
                       (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        
        return annotated_image, enhanced_detections
    
    def _apply_enhanced_nms(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Enhanced NMS với class-aware filtering
        """
        if not detections:
            return []
        
        # Group by class for better NMS
        class_groups = {}
        for i, det in enumerate(detections):
            class_name = det['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append((i, det))
        
        final_detections = []
        
        # Apply NMS per class
        for class_name, class_dets in class_groups.items():
            if not class_dets:
                continue
            
            indices, dets = zip(*class_dets)
            boxes = []
            scores = []
            
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                boxes.append([x1, y1, x2, y2])
                scores.append(det['confidence'])
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            # Apply NMS
            try:
                import torchvision
                keep_indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
                keep_indices = keep_indices.numpy()
            except:
                keep_indices = self._simple_nms(boxes, scores, iou_threshold)
            
            # Add kept detections
            for idx in keep_indices:
                final_detections.append(dets[idx])
        
        return final_detections
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """
        Apply Non-Maximum Suppression để loại bỏ overlapping detections
        """
        if not detections:
            return []
        
        # Convert to format for NMS
        boxes = []
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2, y2])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        try:
            import torchvision
            indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
            indices = indices.numpy()
        except:
            # Fallback NMS implementation
            indices = self._simple_nms(boxes, scores, iou_threshold)
        
        # Return filtered detections
        return [detections[i] for i in indices]
    
    def _simple_nms(self, boxes, scores, iou_threshold):
        """Simple NMS implementation"""
        indices = []
        order = scores.argsort()[::-1]
        
        while len(order) > 0:
            i = order[0]
            indices.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(boxes[i][0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i][1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i][2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i][3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area_i + area_others - intersection
            
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU less than threshold
            keep = np.where(iou <= iou_threshold)[0]
            order = order[keep + 1]
        
        return indices
    
    def detect_video(self, video_path: str, output_path: str = None) -> str:
        """
        Detect objects trong video - FIXED VERSION với codec workaround cho Windows
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Tạo output path nếu không có
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.config.OUTPUT_DIR, f"{video_name}_detected.avi")
        
        # Ensure absolute path
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📹 Processing video: {video_path}")
        print(f"📁 Output will be saved to: {output_path}")
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Fix FPS if invalid
        if fps <= 0 or fps > 60:
            fps = 30
            print(f"⚠️ Invalid FPS detected, using default: {fps}")
        
        # Windows codec workaround - try simple codecs first
        codec_attempts = [
            ('MJPG', '.avi'),
            ('DIVX', '.avi'),
            ('XVID', '.avi'),
            ('mp4v', '.mp4')
        ]
        
        out = None
        successful_codec = None
        final_output_path = output_path
        
        # Try each codec
        for codec_name, ext in codec_attempts:
            try:
                # Adjust extension if needed
                if not final_output_path.lower().endswith(ext):
                    final_output_path = os.path.splitext(output_path)[0] + ext
                
                print(f"🔧 Trying codec: {codec_name}")
                
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
                
                if test_out.isOpened():
                    print(f"✅ Codec {codec_name} works!")
                    out = test_out
                    successful_codec = codec_name
                    break
                else:
                    test_out.release()
                    
            except Exception as e:
                print(f"❌ Codec {codec_name} failed: {e}")
                continue
        
        # If all codecs fail, use frame-by-frame method
        if out is None:
            print("⚠️ All video codecs failed. Using frame-by-frame method...")
            cap.release()
            return self._process_video_frames(video_path, output_path)
        
        # Reset video writer for actual processing (important!)
        out.release()
        out = cv2.VideoWriter(final_output_path, cv2.VideoWriter_fourcc(*successful_codec), fps, (width, height))
        
        if not out.isOpened():
            print("❌ Cannot reopen video writer")
            cap.release()
            return self._process_video_frames(video_path, output_path)
        
        print(f"🎬 Processing with codec: {successful_codec}")
        
        frame_count = 0
        frames_written = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Detect objects in frame với tracking
                    annotated_frame, detections = self.detect_image(frame, use_tracking=True)
                    
                    # Ensure correct format
                    if annotated_frame.shape[:2] != (height, width):
                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                    
                    if annotated_frame.dtype != np.uint8:
                        annotated_frame = annotated_frame.astype(np.uint8)
                    
                    # Write frame
                    if out.write(annotated_frame):
                        frames_written += 1
                    else:
                        # Fallback to original frame
                        fallback_frame = cv2.resize(frame, (width, height))
                        out.write(fallback_frame)
                        frames_written += 1
                    
                except Exception as e:
                    # Write original frame on error
                    fallback_frame = cv2.resize(frame, (width, height))
                    out.write(fallback_frame)
                    frames_written += 1
                
                frame_count += 1
                
                # Progress reporting
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Written: {frames_written}")
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        finally:
            cap.release()
            if out:
                out.release()
        
        # Verify output
        if os.path.exists(final_output_path):
            file_size = os.path.getsize(final_output_path)
            if file_size > 1000:
                print(f"✅ Video processing completed!")
                print(f"📂 Output: {final_output_path}")
                print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
                print(f"📝 Frames written: {frames_written}")
                return final_output_path
            else:
                print(f"❌ Output file too small, using frame method")
                return self._process_video_frames(video_path, output_path)
        else:
            print(f"❌ Output file not created, using frame method")
            return self._process_video_frames(video_path, output_path)
    
    def _process_video_frames(self, video_path: str, output_path: str) -> str:
        """
        Fallback method: process video frame by frame and create HTML viewer
        """
        print("📸 Processing video frame by frame...")
        
        # Create frames directory
        frames_dir = os.path.splitext(output_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        processed_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process frame
                annotated_frame, detections = self.detect_image(frame)
                
                # Save frame
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                processed_frames.append(frame_filename)
                
            except Exception as e:
                # Save original frame as fallback
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                processed_frames.append(frame_filename)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"⏳ Processing frames: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        
        # Create HTML viewer
        html_file = os.path.join(frames_dir, "viewer.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Traffic Detection Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; background: #222; color: white; }}
        .controls {{ margin: 20px; padding: 20px; background: #333; border-radius: 10px; }}
        button {{ margin: 5px; padding: 10px 20px; font-size: 16px; background: #007acc; color: white; border: none; border-radius: 5px; cursor: pointer; }}
        button:hover {{ background: #005c99; }}
        #frameImage {{ max-width: 90%; height: auto; border: 2px solid #007acc; }}
        .info {{ margin: 20px; font-size: 18px; }}
    </style>
</head>
<body>
    <h1>🚦 Traffic Detection Results</h1>
    <div class="info">
        Video: {os.path.basename(video_path)}<br>
        Frames: {len(processed_frames)} | Size: {width}x{height} | FPS: {fps}
    </div>
    <div class="controls">
        <button onclick="prevFrame()">⏮️ Previous</button>
        <button onclick="togglePlay()" id="playBtn">▶️ Play</button>
        <button onclick="nextFrame()">⏭️ Next</button>
        <br><br>
        <span id="frameInfo">Frame 1 / {len(processed_frames)}</span>
    </div>
    <img id="frameImage" src="frame_000001.jpg" alt="Frame">
    
    <script>
        let currentFrame = 1;
        let totalFrames = {len(processed_frames)};
        let isPlaying = false;
        let playInterval = null;
        
        function updateFrame() {{
            document.getElementById('frameImage').src = `frame_${{currentFrame.toString().padStart(6, '0')}}.jpg`;
            document.getElementById('frameInfo').textContent = `Frame ${{currentFrame}} / ${{totalFrames}}`;
        }}
        
        function nextFrame() {{
            if (currentFrame < totalFrames) {{
                currentFrame++;
                updateFrame();
            }} else if (isPlaying) {{
                currentFrame = 1;
                updateFrame();
            }}
        }}
        
        function prevFrame() {{
            if (currentFrame > 1) {{
                currentFrame--;
                updateFrame();
            }}
        }}
        
        function togglePlay() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                document.getElementById('playBtn').textContent = '▶️ Play';
            }} else {{
                isPlaying = true;
                document.getElementById('playBtn').textContent = '⏸️ Pause';
                playInterval = setInterval(nextFrame, 150);
            }}
        }}
        
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') prevFrame();
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
        }});
    </script>
</body>
</html>""")
        
        print(f"✅ Frame processing completed!")
        print(f"📁 Results saved in: {frames_dir}")
        print(f"📄 Open {html_file} in browser to view results")
        
        return html_file
    
    def detect_realtime(self, camera_id: int = 0):
        """
        Detect realtime từ camera với improved error handling
        """
        # Thử mở camera với các backend khác nhau
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        cap = None
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    print(f"Camera opened successfully with backend: {backend}")
                    break
                else:
                    cap.release()
            except:
                continue
        
        if cap is None or not cap.isOpened():
            print(f"Cannot open camera {camera_id}")
            return
        
        # Set camera properties cho performance tốt hơn
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Press 'q' to quit, 's' to save current frame")
        print("System running with Python 3.13.1 - Vietnamese sign classification disabled")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                try:
                    # Detect
                    annotated_frame, detections = self.detect_image(frame)
                    
                    # Thống kê traffic lights
                    traffic_status = self.traffic_light_detector.get_traffic_light_status(detections)
                    
                    # Thông tin cơ bản
                    cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Hiển thị traffic light status
                    y_offset = 90
                    for color, count in traffic_status.items():
                        if count > 0:
                            if color == 'red':
                                text_color = (0, 0, 255)
                            elif color == 'yellow':
                                text_color = (0, 255, 255)
                            elif color == 'green':
                                text_color = (0, 255, 0)
                            else:
                                text_color = (255, 255, 255)
                            
                            cv2.putText(annotated_frame, f"Đèn {color.upper()}: {count}", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                            y_offset += 25
                    
                    # Hiển thị
                    cv2.imshow('Traffic Detection - Press Q to quit', annotated_frame)
                    
                    frame_count += 1
                    
                    # Clear cache mỗi 100 frames
                    if frame_count % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    cv2.imshow('Traffic Detection - Press Q to quit', frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    save_path = f"captured_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"Frame saved: {save_path}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error in realtime detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")
    
    def detect_from_image_file(self, image_path: str, output_path: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect từ image file - IMAGE IMPORT FUNCTIONALITY
        """
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None, []
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot load image: {image_path}")
                return None, []
            
            print(f"📸 Processing image: {os.path.basename(image_path)}")
            print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
            
            # Detect objects
            annotated_image, detections = self.detect_image(image)
            
            # Print results
            if detections:
                print(f"✅ Found {len(detections)} objects:")
                
                # Group detections by type
                traffic_lights = [d for d in detections if 'traffic_light_color' in d]
                signs = [d for d in detections if 'basic_vietnamese_name' in d]
                vehicles = [d for d in detections if d['class_name'] in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']]
                people = [d for d in detections if d['class_name'] == 'person']
                
                if traffic_lights:
                    print(f"  🚦 Traffic lights: {len(traffic_lights)}")
                    for light in traffic_lights:
                        color = light['traffic_light_color']
                        conf = light['confidence']
                        print(f"    - Đèn {color.upper()}: {conf:.2f}")
                
                if signs:
                    print(f"  🚧 Traffic signs: {len(signs)}")
                    for sign in signs:
                        name = sign['basic_vietnamese_name']
                        conf = sign['confidence']
                        print(f"    - {name}: {conf:.2f}")
                
                if vehicles:
                    print(f"  🚗 Vehicles: {len(vehicles)}")
                    vehicle_count = {}
                    for vehicle in vehicles:
                        vtype = vehicle['class_name']
                        vehicle_count[vtype] = vehicle_count.get(vtype, 0) + 1
                    for vtype, count in vehicle_count.items():
                        print(f"    - {vtype}: {count}")
                
                if people:
                    print(f"  🚶 People: {len(people)}")
            else:
                print("❌ No objects detected")
            
            # Save result if output path provided
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(output_path, annotated_image)
                print(f"💾 Result saved: {output_path}")
            
            return annotated_image, detections
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None, []
    
    def analyze_image_batch(self, folder_path: str, output_folder: str = "results"):
        """
        Analyze multiple images from folder - BATCH PROCESSING
        """
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            image_files.extend(glob.glob(pattern.upper()))
        
        if not image_files:
            print(f"❌ No image files found in: {folder_path}")
            print(f"Supported formats: {', '.join(image_extensions)}")
            return
        
        print(f"📁 Found {len(image_files)} images to process")
        print(f"📂 Output folder: {output_folder}")
        
        # Process images
        results_summary = []
        successful = 0
        
        for i, image_path in enumerate(image_files):
            print(f"\n📸 Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Generate output path
            output_name = f"result_{os.path.basename(image_path)}"
            output_path = os.path.join(output_folder, output_name)
            
            # Process image
            annotated_image, detections = self.detect_from_image_file(image_path, output_path)
            
            if annotated_image is not None:
                successful += 1
                
                # Create summary
                summary = {
                    'filename': os.path.basename(image_path),
                    'total_objects': len(detections),
                    'traffic_lights': len([d for d in detections if 'traffic_light_color' in d]),
                    'signs': len([d for d in detections if 'basic_vietnamese_name' in d]),
                    'vehicles': len([d for d in detections if d['class_name'] in ['car', 'truck', 'bus', 'motorcycle']]),
                    'people': len([d for d in detections if d['class_name'] == 'person'])
                }
                results_summary.append(summary)
        
        # Save summary report
        self.save_batch_report(results_summary, output_folder)
        
        print(f"\n✅ Batch processing completed!")
        print(f"📊 Successfully processed: {successful}/{len(image_files)} images")
        print(f"📁 Results saved in: {output_folder}")
    
    def save_batch_report(self, results: List[Dict], output_folder: str):
        """Save batch processing report"""
        report_path = os.path.join(output_folder, "detection_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VIETNAM TRAFFIC DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            total_files = len(results)
            total_objects = sum(r['total_objects'] for r in results)
            total_lights = sum(r['traffic_lights'] for r in results)
            total_signs = sum(r['signs'] for r in results)
            total_vehicles = sum(r['vehicles'] for r in results)
            total_people = sum(r['people'] for r in results)
            
            f.write(f"SUMMARY:\n")
            f.write(f"  Files processed: {total_files}\n")
            f.write(f"  Total objects: {total_objects}\n")
            f.write(f"  Traffic lights: {total_lights}\n")
            f.write(f"  Traffic signs: {total_signs}\n")
            f.write(f"  Vehicles: {total_vehicles}\n")
            f.write(f"  People: {total_people}\n\n")
            
            f.write(f"DETAILED RESULTS:\n")
            f.write(f"{'-'*50}\n")
            
            for result in results:
                f.write(f"\nFile: {result['filename']}\n")
                f.write(f"  Total objects: {result['total_objects']}\n")
                f.write(f"  Traffic lights: {result['traffic_lights']}\n")
                f.write(f"  Traffic signs: {result['signs']}\n")
                f.write(f"  Vehicles: {result['vehicles']}\n")
                f.write(f"  People: {result['people']}\n")
        
        print(f"📄 Report saved: {report_path}")
    
    def display_image_with_detections(self, image: np.ndarray, detections: List[Dict], 
                                    window_name: str = "Traffic Detection Results"):
        """
        Display image với detections - INTERACTIVE VIEWING
        """
        if image is None:
            print("❌ No image to display")
            return
        
        # Resize for display if too large
        display_image = image.copy()
        height, width = display_image.shape[:2]
        
        # Scale down if image is too large
        max_size = 1200
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        # Add detection count to title
        total_objects = len(detections)
        traffic_lights = len([d for d in detections if 'traffic_light_color' in d])
        signs = len([d for d in detections if 'basic_vietnamese_name' in d])
        
        title = f"{window_name} - {total_objects} objects ({traffic_lights} lights, {signs} signs)"
        
        # Display image
        cv2.imshow(title, display_image)
        
        print(f"\n📺 Displaying results...")
        print(f"Press any key to close, 's' to save, ESC to exit")
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            # Save image
            output_path = f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(output_path, display_image)
            print(f"💾 Image saved: {output_path}")
        
        cv2.destroyAllWindows()
        return key != 27  # Return False if ESC pressed

    def test_camera(self, camera_id: int = 0) -> bool:
        """
        Test camera functionality
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except:
            return False