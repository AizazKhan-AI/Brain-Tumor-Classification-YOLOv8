from ultralytics import YOLO

def main():
    # Path to your best model weights
    model_path = r"C:\Users\ncai_4\Desktop\braintumor4classes\runs\classify\train14\weights\best.pt"
    
    # Path to your main dataset folder (YOLO will look for the 'test' folder inside)
    data_path = r"C:\Users\ncai_4\Desktop\braintumor4classes"

    # Load your model
    model = YOLO(model_path)

    # Run validation on the TEST split
    # This command handles subfolders automatically and calculates all matrices
    metrics = model.val(data=data_path, split='test')

    print("\n✅ Evaluation Finished!")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Check your 'runs/classify/val' folder for the Confusion Matrix and Metrics!")

if __name__ == "__main__":
    main()