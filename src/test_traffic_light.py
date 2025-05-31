import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from traffic_light_detector import TrafficLightColorDetector

def test_traffic_light_detection():
    """Test traffic light color detection với webcam"""
    
    # Initialize detector
    detector = TrafficLightColorDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    print("=== Traffic Light Color Detection Test ===")
    print("Instructions:")
    print("1. Point camera at traffic light")
    print("2. Click and drag to select traffic light area")
    print("3. Press 'r' to reset selection")
    print("4. Press 'q' to quit")
    print("=" * 45)
    
    # Mouse callback variables
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, fx, fy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                fx, fy = x, y
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
    
    cv2.namedWindow('Traffic Light Test')
    cv2.setMouseCallback('Traffic Light Test', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        # Draw selection rectangle
        if ix != -1 and iy != -1:
            cv2.rectangle(display_frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
            
            # Detect color if rectangle is drawn
            if abs(fx - ix) > 20 and abs(fy - iy) > 20:
                # Ensure correct order
                x1, x2 = min(ix, fx), max(ix, fx)
                y1, y2 = min(iy, fy), max(iy, fy)
                
                # Detect color
                detected_color = detector.detect_traffic_light_color_advanced(
                    frame, (x1, y1, x2, y2)
                )
                
                # Display result
                label = f"Color: {detected_color.upper()}"
                cv2.putText(display_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Color indicator
                color_map = {
                    'red': (0, 0, 255),
                    'yellow': (0, 255, 255),
                    'green': (0, 255, 0),
                    'unknown': (128, 128, 128)
                }
                
                indicator_color = color_map.get(detected_color, (128, 128, 128))
                cv2.circle(display_frame, (50, 50), 25, indicator_color, -1)
                cv2.circle(display_frame, (50, 50), 25, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "Select traffic light area", 
                   (10, display_frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'r' to reset, 'q' to quit", 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Traffic Light Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            ix, iy, fx, fy = -1, -1, -1, -1
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_image(image_path):
    """Test với ảnh có sẵn"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    detector = TrafficLightColorDetector()
    image = cv2.imread(image_path)
    
    print(f"Testing with image: {image_path}")
    print("Click and drag to select traffic light area")
    
    # Mouse callback for image
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, fx, fy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                fx, fy = x, y
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            
            # Detect color
            if abs(fx - ix) > 10 and abs(fy - iy) > 10:
                x1, x2 = min(ix, fx), max(ix, fx)
                y1, y2 = min(iy, fy), max(iy, fy)
                
                detected_color = detector.detect_traffic_light_color_advanced(
                    image, (x1, y1, x2, y2)
                )
                print(f"Detected color: {detected_color}")
    
    cv2.namedWindow('Image Test')
    cv2.setMouseCallback('Image Test', mouse_callback)
    
    while True:
        display_image = image.copy()
        
        if ix != -1 and iy != -1:
            cv2.rectangle(display_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
        
        cv2.imshow('Image Test', display_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test với ảnh
        test_with_image(sys.argv[1])
    else:
        # Test với camera
        test_traffic_light_detection()