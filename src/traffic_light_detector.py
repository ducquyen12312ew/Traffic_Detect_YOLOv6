import cv2
import numpy as np
from typing import Tuple, List, Dict
import colorsys

class TrafficLightColorDetector:
    def __init__(self):
        # Enhanced HSV ranges cho traffic lights
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'yellow': [
                (np.array([15, 100, 100]), np.array([35, 255, 255]))
            ],
            'green': [
                (np.array([40, 100, 100]), np.array([80, 255, 255]))
            ]
        }
        
        # Enhanced thresholds
        self.min_area_threshold = 0.008
        self.min_brightness = 80
        
    def detect_arrow_enhanced(self, roi: np.ndarray, debug=False) -> str:
        """
        Enhanced arrow detection với multiple methods
        """
        if roi is None or roi.size == 0:
            return 'none'
        
        try:
            height, width = roi.shape[:2]
            
            if debug:
                print(f"🔍 ROI size: {width}x{height}")
            
            # Method 1: Template matching
            template_result = self._detect_arrow_template(roi, debug)
            if template_result != 'none':
                if debug:
                    print(f"✅ Template method: {template_result}")
                return template_result
            
            # Method 2: Brightness distribution analysis
            brightness_result = self._detect_arrow_brightness(roi, debug)
            if brightness_result != 'none':
                if debug:
                    print(f"✅ Brightness method: {brightness_result}")
                return brightness_result
            
            # Method 3: Contour shape analysis
            contour_result = self._detect_arrow_contour(roi, debug)
            if contour_result != 'none':
                if debug:
                    print(f"✅ Contour method: {contour_result}")
                return contour_result
            
            # Method 4: Edge analysis
            edge_result = self._detect_arrow_edges(roi, debug)
            if edge_result != 'none':
                if debug:
                    print(f"✅ Edge method: {edge_result}")
                return edge_result
            
            if debug:
                print("❌ No arrow detected by any method")
            
            return 'none'
            
        except Exception as e:
            if debug:
                print(f"❌ Arrow detection error: {e}")
            return 'none'
    
    def _detect_arrow_template(self, roi: np.ndarray, debug=False) -> str:
        """Template matching for arrows"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Get bright areas
            _, thresh = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)
            
            h, w = thresh.shape
            if w < 15 or h < 15:
                return 'none'
            
            # Create arrow templates
            template_size = min(w//2, h//2, 30)
            
            # Right arrow template
            right_template = np.zeros((template_size, template_size), dtype=np.uint8)
            points = np.array([
                [template_size//4, template_size//4],
                [3*template_size//4, template_size//2],
                [template_size//4, 3*template_size//4]
            ], np.int32)
            cv2.fillPoly(right_template, [points], 255)
            
            # Left arrow template
            left_template = np.zeros((template_size, template_size), dtype=np.uint8)
            points = np.array([
                [3*template_size//4, template_size//4],
                [template_size//4, template_size//2],
                [3*template_size//4, 3*template_size//4]
            ], np.int32)
            cv2.fillPoly(left_template, [points], 255)
            
            # Resize thresh for matching
            thresh_resized = cv2.resize(thresh, (template_size*2, template_size*2))
            
            # Template matching
            right_match = cv2.matchTemplate(thresh_resized, right_template, cv2.TM_CCOEFF_NORMED)
            left_match = cv2.matchTemplate(thresh_resized, left_template, cv2.TM_CCOEFF_NORMED)
            
            _, right_max, _, _ = cv2.minMaxLoc(right_match)
            _, left_max, _, _ = cv2.minMaxLoc(left_match)
            
            if debug:
                print(f"🎯 Template scores - Right: {right_max:.3f}, Left: {left_max:.3f}")
            
            # Lower threshold for better detection
            if right_max > 0.25 and right_max > left_max * 1.2:
                return 'right'
            elif left_max > 0.25 and left_max > right_max * 1.2:
                return 'left'
            
            return 'none'
            
        except Exception as e:
            if debug:
                print(f"Template error: {e}")
            return 'none'
    
    def _detect_arrow_brightness(self, roi: np.ndarray, debug=False) -> str:
        """Brightness distribution analysis"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Get bright areas only
            _, bright_mask = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
            
            if np.sum(bright_mask) < 30:
                return 'none'
            
            h, w = bright_mask.shape
            
            # Horizontal and vertical profiles
            h_profile = np.sum(bright_mask, axis=0)
            v_profile = np.sum(bright_mask, axis=1)
            
            # Calculate center of mass
            if np.sum(h_profile) > 0:
                h_indices = np.arange(len(h_profile))
                h_center = np.sum(h_indices * h_profile) / np.sum(h_profile)
                h_shift = h_center - w/2
            else:
                h_shift = 0
            
            if np.sum(v_profile) > 0:
                v_indices = np.arange(len(v_profile))
                v_center = np.sum(v_indices * v_profile) / np.sum(v_profile)
                v_shift = v_center - h/2
            else:
                v_shift = 0
            
            if debug:
                print(f"📊 Brightness shifts - H: {h_shift:.2f}, V: {v_shift:.2f}")
            
            # Decision thresholds
            h_threshold = w * 0.12
            v_threshold = h * 0.12
            
            if abs(h_shift) > h_threshold:
                if h_shift > 0:
                    return 'right'
                else:
                    return 'left'
            elif abs(v_shift) > v_threshold and abs(h_shift) < h_threshold:
                return 'straight'
            
            return 'none'
            
        except Exception as e:
            if debug:
                print(f"Brightness error: {e}")
            return 'none'
    
    def _detect_arrow_contour(self, roi: np.ndarray, debug=False) -> str:
        """Contour-based arrow detection"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance and threshold
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 110, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 'none'
            
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area < 50:
                return 'none'
            
            # Get bounding box and analyze shape
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = w / h if h > 0 else 1
            
            # Calculate moments for centroid
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                roi_center_x = roi.shape[1] // 2
                
                if debug:
                    print(f"🔺 Contour - Area: {area:.0f}, Aspect: {aspect_ratio:.2f}, CenterX: {cx}/{roi_center_x}")
                
                # Arrow detection based on shape and position
                if aspect_ratio > 1.3:  # Wide shape - horizontal arrow
                    center_shift = cx - roi_center_x
                    if abs(center_shift) > roi.shape[1] * 0.1:
                        if center_shift > 0:
                            return 'right'
                        else:
                            return 'left'
                elif aspect_ratio < 0.8:  # Tall shape - vertical arrow
                    return 'straight'
            
            return 'none'
            
        except Exception as e:
            if debug:
                print(f"Contour error: {e}")
            return 'none'
    
    def _detect_arrow_edges(self, roi: np.ndarray, debug=False) -> str:
        """Edge-based arrow detection"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Analyze edge distribution
            h, w = edges.shape
            
            # Split into regions
            left_region = edges[:, :w//3]
            center_region = edges[:, w//3:2*w//3]
            right_region = edges[:, 2*w//3:]
            
            left_edges = np.sum(left_region)
            center_edges = np.sum(center_region)
            right_edges = np.sum(right_region)
            
            total_edges = left_edges + center_edges + right_edges
            
            if total_edges < 50:
                return 'none'
            
            if debug:
                print(f"🔲 Edge distribution - L: {left_edges}, C: {center_edges}, R: {right_edges}")
            
            # Arrow detection based on edge distribution
            if right_edges > left_edges * 1.5 and right_edges > center_edges:
                return 'right'
            elif left_edges > right_edges * 1.5 and left_edges > center_edges:
                return 'left'
            elif center_edges > max(left_edges, right_edges) * 1.2:
                return 'straight'
            
            return 'none'
            
        except Exception as e:
            if debug:
                print(f"Edge error: {e}")
            return 'none'
    
    def detect_traffic_light_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Basic color detection"""
        x1, y1, x2, y2 = bbox
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
        if max(color_scores.values()) > 0.01:
            detected_color = max(color_scores, key=color_scores.get)
            return detected_color
        
        return 'unknown'
    
    def detect_traffic_light_color_advanced(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Advanced color detection với brightness analysis"""
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
        
        # Multiple brightness thresholds
        bright_masks = []
        for threshold in [self.min_brightness, self.min_brightness + 30, self.min_brightness + 60]:
            _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            bright_masks.append(bright_mask)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Enhanced color detection
        color_brightness_scores = {}
        
        for color_name, ranges in self.color_ranges.items():
            max_score = 0
            
            for bright_mask in bright_masks:
                color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                for lower, upper in ranges:
                    mask = cv2.inRange(hsv, lower, upper)
                    color_mask = cv2.bitwise_or(color_mask, mask)
                
                combined_mask = cv2.bitwise_and(color_mask, bright_mask)
                area = cv2.countNonZero(combined_mask)
                
                roi_area = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
                score = area / roi_area if roi_area > 0 else 0
                
                if area > 0:
                    avg_brightness = np.mean(gray[combined_mask > 0]) / 255.0
                    score *= (1 + avg_brightness)
                
                max_score = max(max_score, score)
            
            color_brightness_scores[color_name] = max_score
        
        # Find dominant color
        if max(color_brightness_scores.values()) > self.min_area_threshold:
            detected_color = max(color_brightness_scores, key=color_brightness_scores.get)
            
            sorted_scores = sorted(color_brightness_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.5:
                return detected_color
            elif sorted_scores[0] > 0.05:
                return detected_color
        
        return 'unknown'
    
    def detect_traffic_light_color_with_arrow(self, image: np.ndarray, bbox: Tuple[int, int, int, int], debug=False) -> Tuple[str, str]:
        """
        Combined color and arrow detection
        """
        # Detect color first
        color = self.detect_traffic_light_color_advanced(image, bbox)
        
        # Only check for arrows in green lights
        arrow = 'none'
        if color == 'green':
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2]
            
            if debug:
                print(f"\n🔍 ARROW DETECTION DEBUG")
                print(f"📍 ROI: ({x1},{y1}) to ({x2},{y2}), Size: {roi.shape}")
            
            arrow = self.detect_arrow_enhanced(roi, debug=debug)
            
            if debug:
                print(f"🎯 Final result - Color: {color}, Arrow: {arrow}")
        
        return color, arrow

class EnhancedTrafficLightDetector:
    def __init__(self):
        self.color_detector = TrafficLightColorDetector()
        
    def analyze_traffic_lights(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Enhanced traffic light analysis với arrow detection
        """
        enhanced_detections = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Check if it's a traffic light
            is_traffic_light = any([
                'traffic' in class_name and 'light' in class_name,
                'traffic_light' in class_name,
                class_name == 'traffic light',
                class_name.startswith('traffic_light_'),
                'light' in class_name and len(class_name.split()) <= 2,
                class_name in ['traffic_signal', 'stoplight', 'signal']
            ])
            
            if is_traffic_light:
                bbox = detection['bbox']
                
                # Enhanced detection với debug mode (có thể bật debug=True để test)
                detected_color, arrow_direction = self.color_detector.detect_traffic_light_color_with_arrow(
                    image, bbox, debug=False
                )
                
                enhanced_detection = detection.copy()
                enhanced_detection['traffic_light_color'] = detected_color
                enhanced_detection['arrow_direction'] = arrow_direction
                enhanced_detection['is_traffic_light'] = True
                
                # Set Vietnamese name based on detection
                if detected_color == 'green' and arrow_direction == 'right':
                    enhanced_detection['vietnamese_name'] = 'DEN_DUOC_PHEP_RE'
                    enhanced_detection['class_name'] = 'traffic_light_green_right'
                elif detected_color == 'green' and arrow_direction == 'left':
                    enhanced_detection['vietnamese_name'] = 'DEN_DUOC_PHEP_RE_TRAI'
                    enhanced_detection['class_name'] = 'traffic_light_green_left'
                elif detected_color == 'green' and arrow_direction == 'straight':
                    enhanced_detection['vietnamese_name'] = 'DEN_DUOC_PHEP_THANG'
                    enhanced_detection['class_name'] = 'traffic_light_green_straight'
                elif detected_color == 'red':
                    enhanced_detection['vietnamese_name'] = 'Den_Do'
                    enhanced_detection['class_name'] = 'traffic_light_red'
                elif detected_color == 'yellow':
                    enhanced_detection['vietnamese_name'] = 'Den_Vang'
                    enhanced_detection['class_name'] = 'traffic_light_yellow'
                elif detected_color == 'green':
                    enhanced_detection['vietnamese_name'] = 'Den_Xanh'
                    enhanced_detection['class_name'] = 'traffic_light_green'
                else:
                    enhanced_detection['vietnamese_name'] = 'Den_Khong_Ro'
                    enhanced_detection['class_name'] = 'traffic_light_unknown'
                
                # Boost confidence for successful detection
                if detected_color != 'unknown':
                    confidence_boost = 0.1
                    if arrow_direction != 'none':
                        confidence_boost += 0.05
                    enhanced_detection['confidence'] = min(1.0, detection['confidence'] + confidence_boost)
                
                enhanced_detections.append(enhanced_detection)
            else:
                enhanced_detections.append(detection)
        
        return enhanced_detections
    
    def get_traffic_light_status(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get traffic light status với arrow information
        """
        status = {
            'red': 0, 
            'yellow': 0, 
            'green': 0, 
            'green_right': 0,
            'green_left': 0,
            'green_straight': 0,
            'unknown': 0
        }
        
        for detection in detections:
            if detection.get('is_traffic_light', False):
                color = detection.get('traffic_light_color', 'unknown')
                arrow = detection.get('arrow_direction', 'none')
                
                if color == 'green' and arrow == 'right':
                    status['green_right'] += 1
                elif color == 'green' and arrow == 'left':
                    status['green_left'] += 1
                elif color == 'green' and arrow == 'straight':
                    status['green_straight'] += 1
                elif color in ['red', 'yellow', 'green']:
                    status[color] += 1
                else:
                    status['unknown'] += 1
        
        return status