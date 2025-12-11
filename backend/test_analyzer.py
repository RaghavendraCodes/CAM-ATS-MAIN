import sys
import os
import cv2
import numpy as np

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server import AttentionAnalyzer

def test_analyzer():
    print("Initializing AttentionAnalyzer...")
    try:
        analyzer = AttentionAnalyzer()
        print("Initialization successful.")
        
        if analyzer.yolo_model:
            print("YOLO model loaded successfully.")
        else:
            print("YOLO model failed to load (check path or dependencies).")
            
        # Create a dummy black image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("Running analysis on dummy image...")
        results = analyzer.analyze_image(image)
        print("Analysis results:", results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyzer()
