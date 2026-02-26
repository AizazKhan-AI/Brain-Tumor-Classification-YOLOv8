from ultralytics import YOLO

def main():
    # Load your best weights
    model = YOLO(r"C:\Users\ncai_4\Desktop\braintumor4classes\runs\classify\train14\weights\best.pt")

    # Path to the NEW dataset folder
    new_dataset_path = r"C:\Users\ncai_4\Desktop\classification_task"

    # --- Test on the 'test' folder ---
    print("\n--- Evaluating TEST folder ---")
    test_metrics = model.val(data=new_dataset_path, split='test')
    
    # --- Test on the 'train' folder ---
    # We tell YOLO to treat the 'train' folder as a validation set just for this test
    print("\n--- Evaluating TRAIN folder ---")
    train_metrics = model.val(data=new_dataset_path, split='train')

    print(f"\n✅ Test Folder Accuracy: {test_metrics.top1:.4f}")
    print(f"✅ Train Folder Accuracy: {train_metrics.top1:.4f}")

if __name__ == "__main__":
    main()