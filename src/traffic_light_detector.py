import cv2
import numpy as np
from typing import Tuple, List, Dict
import colorsys

class TrafficLightColorDetector:
    def __init__(self):
        # Enhanced HSV ranges cho traffic lights - fix để detect tốt hơn
        self.color_ranges = {
            'red': [
                # Red range 1 (lower red) - expanded range
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                # Red range 2 (upper red) - expanded range
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'yellow': [
                # Yellow/amber range - expanded for traffic lights
                (np.array([15, 100, 100]), np.array([35, 255, 255]))
            ],
            'green': [
                # Green range - tuned for traffic lights
                (np.array([40, 100, 100]), np.array([80, 255, 255]))
            ]
        }
        
        # Enhanced thresholds
        self.min_area_threshold = 0.008  # 0.8% of bounding box area
        self.min_brightness = 80  # Lower threshold cho dim lights
        
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
        Advanced color detection với brightness analysis - FIXED VERSION
        """
        x1, y1, x2, y2 = bbox
        traffic_light_roi = image[y1:y2, x1:x2]
        
        if traffic_light_roi.size == 0:
            return 'unknown'
        
        # Enhanced preprocessing
        height, width = traffic_light_roi.shape[:2]
        if height < 40 or width < 40:
            scale = max(40/height, 40/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            traffic_light_roi = cv2.resize(traffic_light_roi, (new_width, new_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(traffic_light_roi, (5, 5), 0)
        
        # Enhanced brightness detection
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Multiple brightness thresholds để catch dim lights
        bright_masks = []
        for threshold in [self.min_brightness, self.min_brightness + 30, self.min_brightness + 60]:
            _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            bright_masks.append(bright_mask)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Enhanced color detection với multiple brightness levels
        color_brightness_scores = {}
        
        for color_name, ranges in self.color_ranges.items():
            max_score = 0
            
            # Try different brightness levels
            for bright_mask in bright_masks:
                color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                for lower, upper in ranges:
                    mask = cv2.inRange(hsv, lower, upper)
                    color_mask = cv2.bitwise_or(color_mask, mask)
                
                # Combine với brightness mask
                combined_mask = cv2.bitwise_and(color_mask, bright_mask)
                area = cv2.countNonZero(combined_mask)
                
                roi_area = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
                score = area / roi_area if roi_area > 0 else 0
                
                # Weight by average brightness in detected region
                if area > 0:
                    avg_brightness = np.mean(gray[combined_mask > 0]) / 255.0
                    score *= (1 + avg_brightness)  # Boost score for brighter regions
                
                max_score = max(max_score, score)
            
            color_brightness_scores[color_name] = max_score
        
        # Find dominant color với lower threshold
        if max(color_brightness_scores.values()) > self.min_area_threshold:
            detected_color = max(color_brightness_scores, key=color_brightness_scores.get)
            
            # Additional validation - check if score is significantly higher than others
            sorted_scores = sorted(color_brightness_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.5:
                return detected_color
            elif sorted_scores[0] > 0.05:  # 5% threshold for single dominant color
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