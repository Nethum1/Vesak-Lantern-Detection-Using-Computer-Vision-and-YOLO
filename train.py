import os
from ultralytics import YOLO


def main():

    # Get the current project directory
    project_dir = os.getcwd()

    # Path to your dataset configuration file
    yaml_path = os.path.join(project_dir, "dataset.yaml")

    print("==============================================")
    print("     VESAK LANTERN YOLO TRAINING")
    print("==============================================")

    print("Ultralytics version: 8.4.x")
    print("Dataset:", yaml_path)

    # Load the YOLOv8 Nano pretrained model
    model = YOLO("yolov8n.pt")

    print("\n--- Starting Custom Model Training ---")

    # Train the model
    results = model.train(

        # Your dataset configuration file
        data=yaml_path,

        # Number of complete passes through the dataset
        epochs=100,

        # Image resolution used during training
        imgsz=640,

        # Number of images processed at once
        batch=8,

        # CPU training
        device="cpu",

        # Number of data-loading workers
        workers=2,

        # Name of the training folder
        name="vesak_lantern_yolov8",

        # Save the best model
        save=True,

        # Validate the model after each epoch
        val=True,

        # Use pretrained weights
        pretrained=True,

        # Early stopping if validation stops improving
        patience=30,

        # Cache images in RAM if enough memory is available
        cache=False

    )

    print("\n==============================================")
    print("     TRAINING COMPLETED SUCCESSFULLY")
    print("==============================================")

    print("\nYour trained model should be here:")
    print("runs/detect/vesak_lantern_yolov8/weights/best.pt")


if __name__ == "__main__":
    main()