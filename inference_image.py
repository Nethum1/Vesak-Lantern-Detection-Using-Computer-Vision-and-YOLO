# inference_image.py
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def run_image_prediction():
    # Hide the main Tkinter window (we only want the file dialog popup)
    root = tk.Tk()
    root.withdraw()

    # 1. Dynamically find the path to your trained model weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "runs", "detect", "vesak_lantern_yolov8", "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}\nDid you finish training?")
        
    print("Loading custom Vesak Lantern model...")
    model = YOLO(weights_path)
    
    # 2. Open a file picker window so you can upload your image
    print("\nPlease select an image file from the popup window...")
    image_path = filedialog.askopenfilename(
        title="Upload Vesak Lantern Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
    )
    
    if not image_path:
        print("No image selected. Exiting execution loop.")
        return

    print(f"\nProcessing target image: {image_path}")
    
    # 3. Run prediction on the uploaded image
    # conf=0.5 ensures it only displays matches it is more than 50% confident about
    results = model.predict(source=image_path, conf=0.5, verbose=False)
    
    # 4. Check results and explicitly print predictions to the terminal
    for r in results:
        boxes = r.boxes
        num_lanterns = len(boxes)
        
        print("\n" + "="*50)
        print("                  DETECTION REPORT                  ")
        print("="*50)
        
        if num_lanterns > 0:
            print(f" RESULT: SUCCESS! Found {num_lanterns} Vesak Lantern(s) in the image.")
            # Loop through each bounding box to print individual confidence scores
            for index, box in enumerate(boxes):
                confidence = float(box.conf[0]) * 100
                print(f"   -> Lantern #{index + 1}: Detected with {confidence:.2f}% confidence.")
        else:
            print(" RESULT: NO Vesak Lantern detected in this image.")
        
        print("="*50 + "\n")
        
        # 5. Render the visual frame with bounding boxes and labels
        annotated_frame = r.plot()
        
        # Display the output window on your monitor screen
        cv2.imshow("Vesak Lantern Detection Result", annotated_frame)
        
        # Automatically save the result image to your Desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_file_path = os.path.join(desktop_path, "vesak_lantern_prediction_output.jpg")
        cv2.imwrite(output_file_path, annotated_frame)
        print(f"Visual window opened. Output image saved safely to your desktop at:\n{output_file_path}\n")
        
        print("Press any key inside the image window to close it.")
        cv2.waitKey(0)  # Wait indefinitely until you press a key
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_image_prediction()