#!/usr/bin/env python3
"""
Main script for Traffic Detection System
Updated với error handling và camera testing
"""

import argparse
import os
import sys
import traceback
from src.detect import TrafficDetector
from src.utils import create_directories
from src.config import Config

def test_installation():
    """Test cài đặt và dependencies"""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO installed")
    except ImportError:
        print("✗ Ultralytics not installed")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Traffic Detection System')
    parser.add_argument('--mode', type=str, 
                       choices=['video', 'image', 'realtime', 'setup', 'test', 'test-camera'],
                       default='video', help='Detection mode')
    parser.add_argument('--input', type=str, help='Input video/image path')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--model', type=str, help='Custom YOLO model path')
    parser.add_argument('--sign-model', type=str, help='Vietnamese sign CNN model path')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for realtime')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        print("Setting up project directories...")
        create_directories()
        print("Setup completed!")
        return
    
    if args.mode == 'test':
        print("Testing installation...")
        if test_installation():
            print("✓ All dependencies installed correctly!")
        else:
            print("✗ Some dependencies missing. Please install requirements.")
        return
    
    if args.mode == 'test-camera':
        print(f"Testing camera {args.camera}...")
        try:
            detector = TrafficDetector()
            if detector.test_camera(args.camera):
                print("✓ Camera test successful!")
            else:
                print("✗ Camera test failed!")
        except Exception as e:
            print(f"✗ Camera test error: {e}")
        return
    
    try:
        # Khởi tạo detector
        print("Initializing detector...")
        detector = TrafficDetector(args.model, args.sign_model)
        
        if args.mode == 'video':
            if not args.input:
                print("Please provide input video path with --input")
                return
            
            if not os.path.exists(args.input):
                print(f"Input video file not found: {args.input}")
                return
                
            print(f"Processing video: {args.input}")
            output_path = detector.detect_video(args.input, args.output)
            print(f"✓ Video processing completed: {output_path}")
            
        elif args.mode == 'image':
            if not args.input:
                print("Please provide input image path with --input")
                return
            
            if not os.path.exists(args.input):
                print(f"Input image file not found: {args.input}")
                return
            
            import cv2
            image = cv2.imread(args.input)
            if image is None:
                print(f"Could not load image: {args.input}")
                return
                
            print(f"Processing image: {args.input}")
            annotated_image, detections = detector.detect_image(image)
            
            print(f"✓ Found {len(detections)} objects")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            
            # Hiển thị kết quả
            cv2.imshow('Detection Result - Press any key to close', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Lưu nếu có output path
            if args.output:
                cv2.imwrite(args.output, annotated_image)
                print(f"✓ Result saved to: {args.output}")
                
        elif args.mode == 'realtime':
            print(f"Starting realtime detection with camera {args.camera}")
            print("Press 'q' to quit, 's' to save frame")
            detector.detect_realtime(args.camera)
            
    except KeyboardInterrupt:
        print("\n✓ Process interrupted by user")
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        
        # Gợi ý troubleshooting
        print("\n=== Troubleshooting ===")
        if "cuda" in str(e).lower() or "gpu" in str(e).lower():
            print("- Try using CPU: add --device cpu")
            print("- Update GPU drivers")
        elif "camera" in str(e).lower() or "video" in str(e).lower():
            print("- Check camera connection")
            print("- Try different camera ID: --camera 1")
            print("- Test camera first: --mode test-camera")
        elif "model" in str(e).lower():
            print("- Check internet connection for model download")
            print("- Try different model size in config.py")
        else:
            print("- Check if all requirements are installed")
            print("- Run: python main.py --mode test")

if __name__ == "__main__":
    main()