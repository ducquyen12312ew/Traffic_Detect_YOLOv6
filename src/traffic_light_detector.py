import cv2
import numpy as np
from typing import Tuple, List, Dict
import colorsys

class TrafficLightColorDetector:
    def __init__(self):
        # HSV ranges for traffic light colors
        self.color_ranges = {
            'red': [
                # Red range 1 (0-10)
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                # Red range 2 (170-180) 
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'yellow': [
                (np.array([15, 50, 50]), np.array([35, 255, 255]))
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ]
        }
        
        # Minimum area threshold for valid color detection
        self.min_area_threshold = 50
        
    def detect_traffic_light_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Detect color of traffic light trong bounding box
        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) bounding box của traffic light
        Returns:
            color: 'red', 'yellow', 'green', hoặc 'unknown'
        """
        x1, y1, x2, y2 = bbox
        
        # Crop traffic light region
        traffic_light_roi = image[y1:y2, x1:x2]
        
        if traffic_light_roi.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
        
        # Detect each color
        color_scores = {}
        
        for color_name, ranges in self.color_ranges.items():
            total_area = 0
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                area = cv2.countNonZero(mask)
                total_area += area
            
            # Normalize by ROI area
            roi_area = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
            color_scores[color_name] = total_area / roi_area if roi_area > 0 else 0
        
        # Find dominant color
        if max(color_scores.values()) > 0.01:  # Threshold 1%
            detected_color = max(color_scores, key=color_scores.get)
            return detected_color
        
        return 'unknown'
    
    def detect_traffic_light_color_advanced(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Advanced color detection với brightness analysis
        """
        x1, y1, x2, y2 = bbox
        traffic_light_roi = image[y1:y2, x1:x2]
        
        if traffic_light_roi.size == 0:
            return 'unknown'
        
        # Resize ROI for better processing
        height, width = traffic_light_roi.shape[:2]
        if height < 50 or width < 50:
            scale = max(50/height, 50/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            traffic_light_roi = cv2.resize(traffic_light_roi, (new_width, new_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(traffic_light_roi, (5, 5), 0)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Analyze brightness để tìm vùng sáng nhất (đèn đang bật)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Find brightest regions
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Combine brightness with color detection
        color_brightness_scores = {}
        
        for color_name, ranges in self.color_ranges.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                color_mask = cv2.bitwise_or(color_mask, mask)
            
            # Combine với brightness mask
            combined_mask = cv2.bitwise_and(color_mask, bright_mask)
            area = cv2.countNonZero(combined_mask)
            
            roi_area = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
            color_brightness_scores[color_name] = area / roi_area if roi_area > 0 else 0
        
        # Find dominant color với brightness
        if max(color_brightness_scores.values()) > 0.005:  # Threshold 0.5%
            detected_color = max(color_brightness_scores, key=color_brightness_scores.get)
            return detected_color
        
        # Fallback to simple color detection
        return self.detect_traffic_light_color(image, bbox)
    
    def visualize_color_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                 detected_color: str) -> np.ndarray:
        """
        Visualize color detection result
        """
        x1, y1, x2, y2 = bbox
        result_image = image.copy()
        
        # Color mapping
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255), 
            'green': (0, 255, 0),
            'unknown': (128, 128, 128)
        }
        
        color = color_map.get(detected_color, (128, 128, 128))
        
        # Draw bounding box với màu tương ứng
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        label = f"Traffic Light: {detected_color.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background cho text
        cv2.rectangle(result_image, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)
        
        # Text màu trắng
        cv2.putText(result_image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        
        return result_image

class EnhancedTrafficLightDetector:
    def __init__(self):
        self.color_detector = TrafficLightColorDetector()
        
    def analyze_traffic_lights(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Phân tích màu cho tất cả traffic lights được detect
        """
        enhanced_detections = []
        
        for detection in detections:
            if 'traffic' in detection['class_name'].lower() and 'light' in detection['class_name'].lower():
                # Detect color
                bbox = detection['bbox']
                detected_color = self.color_detector.detect_traffic_light_color_advanced(image, bbox)
                
                # Update detection
                enhanced_detection = detection.copy()
                enhanced_detection['traffic_light_color'] = detected_color
                enhanced_detection['class_name'] = f"traffic_light_{detected_color}"
                
                # Adjust confidence based on color detection
                if detected_color != 'unknown':
                    enhanced_detection['confidence'] = min(1.0, detection['confidence'] + 0.1)
                
                enhanced_detections.append(enhanced_detection)
            else:
                enhanced_detections.append(detection)
        
        return enhanced_detections
    
    def get_traffic_light_status(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Thống kê trạng thái đèn giao thông
        """
        status = {'red': 0, 'yellow': 0, 'green': 0, 'unknown': 0}
        
        for detection in detections:
            if 'traffic_light_color' in detection:
                color = detection['traffic_light_color']
                status[color] += 1
        
        return status