Brain Tumor Classification using YOLOv8
📌 Project Overview

This project implements Brain Tumor MRI Image Classification using YOLOv8 Classification Model from Ultralytics.

The model classifies brain MRI images into four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The system is trained using transfer learning on YOLOv8m-cls and evaluated on internal and external datasets.

🚀 Model Architecture

Base Model: YOLOv8 (Classification version)

Pretrained Weights: yolov8m-cls.pt

Framework: PyTorch (via Ultralytics)

Transfer Learning: Yes

GPU Used: RTX 5070 Ti

Image Size: 416 × 416

Epochs: 70

Batch Size: 32

Dropout: 0.3

Early Stopping Patience: 15

## 📂 Project Structure

```
braintumor4classes/
│
├── train/                # Training dataset
├── val/                  # Validation dataset
├── test/                 # Testing dataset
├── External_Dataset/     # External evaluation dataset
├── classification_task/  # Classification experiments
├── runs/                 # Training results & saved weights
│
├── train_yolov8.py       # Main training script
├── test.py               # Testing script
├── README.md
├── LICENSE
└── .gitignore
```
Training Code


from multiprocessing import freeze_support
from ultralytics import YOLO

def main():
    model = YOLO("yolov8m-cls.pt")

    model.train(
        data=r"C:\Users\ncai_4\Desktop\braintumor4classes",
        epochs=70,
        imgsz=416,
        batch=32,
        device=0,
        workers=4,
        patience=15,
        dropout=0.3,
        save=True,
        augment=True
    )

if __name__ == "__main__":
    freeze_support()
    main()
📊 Dataset

The dataset contains MRI brain scans divided into:

Training set

Validation set

Testing set

External dataset (for generalization testing)

⚠️ Note: The dataset is not uploaded to this repository due to size and medical data considerations.

You can use publicly available brain tumor MRI datasets or request access if needed.

📈 Training Results

The model achieves strong classification performance across all four classes.

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Detailed results are available in the /runs directory.

🧪 How to Run
1️⃣ Install Dependencies
pip install ultralytics
2️⃣ Train the Model
python train_yolov8.py
3️⃣ Test the Model
python test.py
💡 Key Features

✔ Transfer Learning with YOLOv8
✔ Multi-class Brain Tumor Classification
✔ External Dataset Evaluation
✔ GPU Accelerated Training
✔ Data Augmentation Enabled
✔ Early Stopping Regularization

🔬 Research Contribution

This project demonstrates:

Application of YOLOv8 for medical image classification

Evaluation on unseen external dataset

Regularization techniques (Dropout + Early Stopping)

Real-world GPU training pipeline

📌 Future Improvements

Hyperparameter optimization

Model comparison (ResNet, EfficientNet, ViT)

Grad-CAM visualization for explainability

Deployment as Web Application

👨‍💻 Author

Muhammad Aizaz
BS Artificial Intelligence , UEAS SWAT 
Brain Tumor Classification Research Project
Contact

📩 Email: muhammadaizaz632@gmail.com

For:

Dataset access

Research collaboration

Model details

Academic discussions
