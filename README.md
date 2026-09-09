# Computing-Research-Project
Explainable Deep Learning for Early Detection and Severity Classification of Diabetic Retinopathy Using the DDR Retinal Image Dataset
# DDR Diabetic Retinopathy — Fast Full Colab Pipeline

This repository contains a streamlined, end-to-end PyTorch pipeline designed for Google Colab to classify Diabetic Retinopathy using the DDR dataset[span_0](start_span)[span_0](end_span). The notebook trains and evaluates multiple transfer-learning models using mixed precision for accelerated training[span_1](start_span)[span_1](end_span).

## Features

* **Data Sampling:** Automatically selects up to 1,000 unique images per class to prevent class imbalance, while keeping classes with fewer than 1,000 images intact without duplication[span_2](start_span)[span_2](end_span).
* **Stratified Splitting:** Creates a highly balanced 70/15/15 split for Training, Validation, and Testing[span_3](start_span)[span_3](end_span).
* **I/O Optimization:** Copies selected images from Google Drive directly to Colab's local storage (`/content/DDR_fast`) to eliminate I/O bottlenecks during training[span_4](start_span)[span_4](end_span).
* **Advanced Preprocessing:** Applies automated fundus cropping, CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement, and resizes images to 224x224[span_5](start_span)[span_5](end_span).
* **Model Zoo:** Trains four distinct architectures with frozen-head initialization followed by full fine-tuning: EfficientNetB0, EfficientNetV2-S, ResNet50, and DenseNet121[span_6](start_span)[span_6](end_span).
* **Comprehensive Evaluation:** Evaluates models using Accuracy, Precision, Recall, F1-score, Multi-class AUC, and Quadratic Weighted Kappa (QWK)[span_7](start_span)[span_7](end_span).
* **Visualizations:** Automatically generates and saves training curves and confusion matrices for all models[span_8](start_span)[span_8](end_span).

## Dataset Directory Structure

For the pipeline to mount and read your dataset correctly in Google Colab, ensure your Google Drive is structured exactly as follows[span_9](start_span)[span_9](end_span):

```text
MyDrive/DDR dataset/
├── DR_grading/
│ └── *.jpg (or .png, .jpeg, .tif, .bmp)
└── DR_grading.csv
