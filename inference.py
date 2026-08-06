# inference.py
import cv2
import os
from ultralytics import YOLO

def run_predictions():
    # Point directly to our freshly generated fine-tuned weights matrix in train2
    weights_path = os.path.join(os.getcwd(), "runs", "detect", "train-2", "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights target error: File not found at {weights_path}")
        
    print(f"Loading custom Vesak Lantern model from: {weights_path}")
    model = YOLO(weights_path)
    
    # 0 targets your default laptop/USB webcam.
    # To test a static image instead, change this to: source = "path_to_your_image.jpg"
    source = 0 
    
    # Run inference with a stream generator
    results = model.predict(source=source, stream=True, conf=0.5)
    
    print("Starting live stream... Press 'q' inside the video window to exit.")
    
    for r in results:
        # Generate the visual frame with bounding boxes and labels
        annotated_frame = r.plot()
        
        # Display the live frame
        cv2.imshow("Vesak Lantern Detection Live Stream", annotated_frame)
        
        # Standard robust OpenCV key check for Windows systems
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or Esc key to quit
            break
            
    # Clean up and release system resources properly
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_predictions()