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
        
    def detect_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects trong một ảnh với error handling
        Returns:
            annotated_image: Ảnh đã được vẽ bounding box
            detections: List các detection
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return image, []
        
        detections = []
        annotated_image = image.copy()
        
        try:
            # Clear cache để tránh memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Detect với error handling
            results = self.model.predict(
                processed_image,
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                verbose=False,
                save=False,
                imgsz=640
            )
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    try:
                        # Lấy thông tin box với error handling
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
                        
                        # Kiểm tra class_id hợp lệ
                        if class_id < len(self.model.names):
                            class_name = self.model.names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        # Kiểm tra coordinates hợp lệ
                        h, w = image.shape[:2]
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))
                        
                        # Lưu detection info
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'class_name': class_name,
                            'confidence': confidence,
                            'class_id': class_id
                        }
                        detections.append(detection)
                        
                    except Exception as e:
                        print(f"Error processing detection {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in detection: {e}")
            return image, []
        
        # Enhance traffic light detections với color analysis
        enhanced_detections = self.traffic_light_detector.analyze_traffic_lights(image, detections)
        
        # Basic sign enhancement (không cần CNN)
        final_detections = []
        for detection in enhanced_detections:
            class_name = detection['class_name'].lower()
            
            # Check if it's a traffic sign
            sign_keywords = ['sign', 'limit', 'stop', 'warning', 'yield', 'no_']
            is_traffic_sign = any(keyword in class_name for keyword in sign_keywords)
            
            if is_traffic_sign and 'traffic_light' not in class_name:
                # Enhanced sign classification với ROI analysis
                roi = image[x1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
                sign_info = self.get_basic_sign_classification(class_name, roi)
                detection['basic_vietnamese_name'] = sign_info['vietnamese_name']
                detection['sign_color'] = sign_info['color']
                detection['confidence'] += sign_info.get('confidence_boost', 0)
                detection['enhanced_sign'] = True
            
            final_detections.append(detection)
        
        # Redraw bounding boxes với enhanced detections
        annotated_image = image.copy()
        for detection in final_detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Special handling cho traffic lights với color
            if 'traffic_light_' in class_name and 'traffic_light_color' in detection:
                color_name = detection['traffic_light_color']
                if color_name == 'red':
                    color = (0, 0, 255)  # Red
                elif color_name == 'yellow':
                    color = (0, 255, 255)  # Yellow
                elif color_name == 'green':
                    color = (0, 255, 0)  # Green
                else:
                    color = (128, 128, 128)  # Gray for unknown
                    
                display_name = f"Đèn {color_name.upper()}"
                
            # Basic Vietnamese signs
            elif 'basic_vietnamese_name' in detection:
                display_name = detection['basic_vietnamese_name']
                color = detection.get('sign_color', (0, 255, 0))
                
            else:
                display_name = class_name
                color = self.config.CLASS_COLORS.get(class_name, (0, 255, 0))
            
            annotated_image = draw_bounding_box(
                annotated_image, (x1, y1, x2, y2),
                display_name, confidence, color
            )
        
        return annotated_image, final_detections
    
    def detect_video(self, video_path: str, output_path: str = None) -> str:
        """
        Detect objects trong video và xuất video kết quả
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Tạo output path nếu không có
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.config.OUTPUT_DIR, f"{video_name}_detected.mp4")
            
        # Mở video với error handling
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Đảm bảo output directory tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Tạo video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.VIDEO_CODEC)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot create video writer for: {output_path}")
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Output: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Detect trong frame
                annotated_frame, detections = self.detect_image(frame)
                
                # Ghi frame
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress mỗi 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                    
                    # Clear cache định kỳ
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            # Cleanup
            cap.release()
            out.release()
        
        print(f"Video processing completed: {output_path}")
        return output_path
    
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
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
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