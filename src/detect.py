import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import os
import torch
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
    
    def get_basic_sign_classification(self, class_name: str) -> Dict:
        """
        Basic classification cho traffic signs mà không cần CNN
        """
        sign_keywords = {
            'stop': {'vietnamese_name': 'Biển báo dừng', 'color': (0, 0, 255)},
            'speed': {'vietnamese_name': 'Biển giới hạn tốc độ', 'color': (255, 0, 0)},
            'yield': {'vietnamese_name': 'Biển nhường đường', 'color': (0, 255, 255)},
            'no_entry': {'vietnamese_name': 'Biển cấm đi ngược chiều', 'color': (0, 0, 255)},
            'warning': {'vietnamese_name': 'Biển cảnh báo', 'color': (0, 255, 255)},
            'mandatory': {'vietnamese_name': 'Biển chỉ dẫn', 'color': (255, 0, 0)},
            'information': {'vietnamese_name': 'Biển thông tin', 'color': (0, 255, 0)}
        }
        
        # Tìm keyword trong class name
        for keyword, info in sign_keywords.items():
            if keyword in class_name.lower():
                return {
                    'vietnamese_name': info['vietnamese_name'],
                    'color': info['color'],
                    'confidence': 0.8
                }
        
        # Default cho unknown signs
        return {
            'vietnamese_name': 'Biển báo giao thông',
            'color': (128, 128, 128),
            'confidence': 0.5
        }
        
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
                # Basic classification
                sign_info = self.get_basic_sign_classification(class_name)
                detection['basic_vietnamese_name'] = sign_info['vietnamese_name']
                detection['sign_color'] = sign_info['color']
            
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
    
    def test_camera(self, camera_id: int = 0):
        """Test camera connection"""
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {camera_id} is working properly")
                print(f"Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print(f"Camera {camera_id} opened but cannot read frames")
                cap.release()
                return False
        else:
            print(f"Cannot open camera {camera_id}")
            return False