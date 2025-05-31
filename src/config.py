import os
import torch

class Config:
    # Đường dẫn
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output_videos')
    
    # Device selection - auto detect
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    DEVICE = 'cpu'  # Force CPU để tránh lỗi GPU
    
    # Model configuration - Dùng model nhỏ hơn để tránh memory issues
    MODEL_SIZE = 'yolov8n'  # yolov8n (smallest), yolov8s, yolov8m, yolov8l, yolov8x
    INPUT_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 8  # Giảm batch size để tránh memory issues
    LEARNING_RATE = 0.01
    
    # Video processing
    VIDEO_CODEC = 'mp4v'
    OUTPUT_FPS = 30
    
    # Memory management
    MAX_IMAGE_SIZE = 1280  # Giới hạn kích thước ảnh input
    CLEAR_CACHE_INTERVAL = 100  # Clear cache mỗi N frames
    
    # Classes colors (BGR format for OpenCV)
    CLASS_COLORS = {
        # Vehicles
        'car': (255, 0, 0),           # Blue
        'truck': (0, 255, 0),         # Green  
        'bus': (0, 0, 255),           # Red
        'motorcycle': (255, 255, 0),  # Cyan
        'bicycle': (255, 0, 255),     # Magenta
        'person': (0, 255, 255),      # Yellow
        
        # Traffic lights - Most important
        'traffic_light_red': (0, 0, 255),      # Red
        'traffic_light_yellow': (0, 255, 255), # Yellow
        'traffic_light_green': (0, 255, 0),    # Green
        
        # Vietnamese Traffic Signs
        'stop_sign': (0, 0, 128),             # Dark Red
        'no_entry': (128, 0, 0),              # Dark Blue
        'no_cars': (0, 128, 0),               # Dark Green
        'no_turn_left': (128, 128, 0),        # Dark Cyan
        'no_motorcycles': (128, 0, 128),      # Dark Magenta
        'no_trucks': (0, 128, 128),           # Dark Yellow
        'weight_limit': (64, 64, 64),         # Dark Gray
        'no_pedestrians': (192, 192, 192),    # Light Gray
        'height_limit': (128, 64, 0),         # Brown
        
        # Speed limits
        'speed_limit_40': (255, 128, 0),      # Orange
        'speed_limit_50': (255, 64, 0),       # Dark Orange
        'speed_limit_60': (255, 0, 128),      # Pink
        
        # Warning signs
        'warning_intersection': (255, 255, 128),     # Light Yellow
        'warning_narrow_road': (255, 128, 128),      # Light Pink
        'warning_intersection_cross': (128, 255, 128), # Light Green
        
        # Mandatory signs
        'yield_sign': (128, 128, 255),        # Light Blue
        'mandatory_straight': (64, 128, 255), # Sky Blue
        'mandatory_roundabout': (255, 128, 64), # Light Orange
        'priority_road': (128, 255, 64),      # Light Green-Yellow
    }
    
    # Vietnamese traffic signs mapping (based on the document)
    VIETNAM_SIGNS = {
        102: 'stop_sign',
        '103a': 'no_entry', 
        '103b': 'no_cars',
        '103c': 'no_turn_left',
        104: 'no_motorcycles',
        '106a': 'no_trucks', 
        '106b': 'weight_limit',
        112: 'no_pedestrians',
        117: 'height_limit',
        '127_40': 'speed_limit_40',
        '127_50': 'speed_limit_50',
        '127_60': 'speed_limit_60',
        201: 'warning_intersection',
        203: 'warning_narrow_road', 
        205: 'warning_intersection_cross',
        208: 'yield_sign',
        301: 'mandatory_straight',
        303: 'mandatory_roundabout',
        401: 'priority_road'
    }
    
    @classmethod
    def create_output_dirs(cls):
        """Tạo các thư mục output cần thiết"""
        dirs = [cls.OUTPUT_DIR, cls.MODEL_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """In ra cấu hình hiện tại"""
        print("=== Traffic Detection Configuration ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_SIZE}")
        print(f"Input size: {cls.INPUT_SIZE}")
        print(f"Confidence threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"IOU threshold: {cls.IOU_THRESHOLD}")
        print(f"Max image size: {cls.MAX_IMAGE_SIZE}")
        print(f"Output directory: {cls.OUTPUT_DIR}")
        print("=" * 40)