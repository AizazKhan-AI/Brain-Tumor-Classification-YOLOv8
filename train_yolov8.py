from multiprocessing import freeze_support
from ultralytics import YOLO

def main():
    # Load pretrained classification model
    model = YOLO("yolov8m-cls.pt") 

    # Train the model
    model.train(
        data=r"C:\Users\ncai_4\Desktop\braintumor4classes",
        epochs=70,
        imgsz=416,       # Sticking with your choice (multiple of 32)
        batch=32,        # Your requested batch size
        device=0,        # Uses your RTX 5070 Ti
        workers=4,
        patience=15,     
        dropout=0.3,
        save=True,
        augment=True
    )

if __name__ == "__main__":
    freeze_support()
    main()