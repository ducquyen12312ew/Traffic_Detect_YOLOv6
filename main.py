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
                       choices=['video', 'image', 'realtime', 'setup', 'test', 'test-camera', 'batch', 'interactive'],
                       default='video', help='Detection mode')
    parser.add_argument('--input', type=str, help='Input video/image/folder path')
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
            
            try:
                output_path = detector.detect_video(args.input, args.output)
                
                if output_path and os.path.exists(output_path):
                    print(f"✓ Video processing completed: {output_path}")
                    
                    # Show results info
                    if output_path.endswith('.html'):
                        print("📄 HTML viewer created - open in browser to view results")
                        print("💡 Use arrow keys to navigate, spacebar to play/pause")
                    elif output_path.endswith('_frames'):
                        print("📁 Frame sequence created - check the folder for processed images")
                        viewer_html = os.path.join(output_path, "viewer.html")
                        if os.path.exists(viewer_html):
                            print(f"📄 HTML viewer available: {viewer_html}")
                    else:
                        print("🎬 Video file created successfully")
                        
                        # Try to get file info
                        try:
                            file_size = os.path.getsize(output_path) / (1024 * 1024)
                            print(f"📂 File size: {file_size:.2f} MB")
                        except:
                            pass
                else:
                    print("❌ Video processing failed")
                    
            except Exception as e:
                print(f"❌ Video processing error: {e}")
                print("\n💡 Troubleshooting:")
                print("1. Check if input video file is valid")
                print("2. Try with a shorter video first")
                print("3. Install K-Lite Codec Pack for better codec support")
                print("4. Use --mode image to test detection on single frames")
            
        elif args.mode == 'image':
            if not args.input:
                print("Please provide input image path with --input")
                return
            
            if not os.path.exists(args.input):
                print(f"Input image file not found: {args.input}")
                return
                
            print(f"🖼️ Processing image: {args.input}")
            annotated_image, detections = detector.detect_from_image_file(args.input, args.output)
            
            if annotated_image is not None:
                # Display results interactively
                detector.display_image_with_detections(annotated_image, detections)
                print("✅ Image processing completed!")
            else:
                print("❌ Failed to process image")
                
        elif args.mode == 'batch':
            if not args.input:
                print("Please provide input folder path with --input")
                return
            
            if not os.path.exists(args.input):
                print(f"Input folder not found: {args.input}")
                return
                
            output_folder = args.output or "batch_results"
            print(f"📁 Processing batch images from: {args.input}")
            detector.analyze_image_batch(args.input, output_folder)
            
        elif args.mode == 'interactive':
            print("🎮 Interactive mode - Process images interactively")
            
            while True:
                print("\n" + "="*50)
                print("INTERACTIVE TRAFFIC DETECTION")
                print("="*50)
                print("1. Process single image")
                print("2. Process image folder (batch)")
                print("3. Start camera detection")
                print("4. Exit")
                
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == '1':
                    image_path = input("Enter image path: ").strip()
                    if os.path.exists(image_path):
                        print(f"Processing: {image_path}")
                        annotated_image, detections = detector.detect_from_image_file(image_path)
                        if annotated_image is not None:
                            continue_viewing = detector.display_image_with_detections(annotated_image, detections)
                            if not continue_viewing:
                                break
                    else:
                        print(f"❌ File not found: {image_path}")
                
                elif choice == '2':
                    folder_path = input("Enter folder path: ").strip()
                    if os.path.exists(folder_path):
                        output_folder = input("Enter output folder (default: batch_results): ").strip() or "batch_results"
                        detector.analyze_image_batch(folder_path, output_folder)
                    else:
                        print(f"❌ Folder not found: {folder_path}")
                
                elif choice == '3':
                    camera_id = input("Enter camera ID (default: 0): ").strip()
                    camera_id = int(camera_id) if camera_id.isdigit() else 0
                    print(f"Starting camera detection (Camera {camera_id})")
                    detector.detect_realtime(camera_id)
                
                elif choice == '4':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice!")
                
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
        elif "image" in str(e).lower() or "file" in str(e).lower():
            print("- Check if image file exists and is readable")
            print("- Supported formats: jpg, jpeg, png, bmp, tiff, webp")
            print("- Try with different image")
        elif "codec" in str(e).lower() or "fourcc" in str(e).lower():
            print("- Video codec issue detected")
            print("- Try different output format (e.g., .avi instead of .mp4)")
            print("- Install additional video codecs")
        else:
            print("- Check if all requirements are installed")
            print("- Run: python main.py --mode test")
        
        print("\n=== Usage Examples ===")
        print("Image detection:")
        print("  python main.py --mode image --input image.jpg")
        print("Batch processing:")
        print("  python main.py --mode batch --input ./images --output ./results")
        print("Interactive mode:")
        print("  python main.py --mode interactive")
        print("Camera detection:")
        print("  python main.py --mode realtime --camera 0")
        print("Video detection:")
        print("  python main.py --mode video --input video.mp4 --output result.mp4")

if __name__ == "__main__":
    main()