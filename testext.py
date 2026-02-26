from ultralytics import YOLO
import os

def main():
    # Load your best trained model
    model = YOLO(r"C:\Users\ncai_4\Desktop\braintumor4classes\runs\classify\train14\weights\best.pt")

    # Path to your COMPLETELY NEW dataset
    new_data_path = r"C:\Users\ncai_4\Desktop\External_Dataset"

    # Run prediction
    # 'save=True' will create a folder with the labels written on the images
    results = model.predict(source=new_data_path, save=True, imgsz=416, conf=0.5)

    print(f"Prediction on new dataset is complete!")
    print(f"Results are saved in: {results[0].save_dir}")

if __name__ == "__main__":
    main()