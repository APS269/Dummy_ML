# Vehicle Detection Pipeline

This repository implements a vehicle detection system using Histogram of Oriented Gradients (HOG) and a Support Vector Machine (SVM). The system can detect vehicles in images or video frames. It utilizes a sliding window approach, HOG feature extraction, SVM classification, and heatmap-based thresholding for robust detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Preprocessing](#preprocessing)
3. [Sliding Window](#sliding-window)
4. [Feature Extraction](#feature-extraction)
5. [Training the Classifier](#training-the-classifier)
6. [Detection Pipeline](#detection-pipeline)
7. [Results](#results)
8. [Usage](#usage)
9. [Dependencies](#dependencies)
10. [License](#license)

---

## Overview

The vehicle detection system uses a series of steps to process images and identify vehicles using HOG features and an SVM classifier:

- **Preprocessing**: Convert image to grayscale or another suitable color space (e.g., YUV or HLS).
- **Sliding Window**: The sliding window moves across the image, and features are extracted from each window.
- **Feature Extraction**: HOG features are calculated from the image windows.
- **Classification**: A trained SVM classifier determines whether the window contains a vehicle.
- **Heatmap Creation**: A heatmap is generated to accumulate detections across frames.
- **Thresholding**: False positives are removed using a threshold on the heatmap.
- **Bounding Boxes**: Bounding boxes are drawn around detected vehicles.

---

## Preprocessing

Before detection begins, the image is preprocessed to ensure optimal feature extraction:

1. Convert the image to grayscale (if not already).
2. Optionally, apply other color space conversions, such as RGB to YUV or HLS, for better feature extraction.

---

## Sliding Window

The sliding window technique is used to scan the image and detect vehicles at different locations and scales. The window moves across the image with a specified overlap (e.g., 50% overlap), ensuring thorough coverage of the image.

- **Window Size**: A typical window size is 64x64 pixels.
- **Overlap**: The window overlaps the previous one by 50% (can be adjusted).
  
For each window, HOG features are extracted, and a classification decision is made.

---

## Feature Extraction

HOG (Histogram of Oriented Gradients) is used to capture edge patterns in an image. It is especially effective for recognizing objects like vehicles, as it focuses on the shape of objects.

The HOG features are calculated as follows:

1. **Orientation Bins**: The gradient orientations within each cell are grouped into bins.
2. **Cells and Blocks**: The image is divided into small cells (e.g., 8x8 pixels). Each block consists of multiple cells, and the gradient information is normalized over these blocks.
3. **Feature Vector**: The final feature vector is formed by concatenating the histograms of all cells across the image.

---

## Training the Classifier

The classifier (SVM) is trained on labeled datasets consisting of images with vehicles and images without vehicles. The steps involved are:

1. **Dataset**: Collect positive (vehicle) and negative (non-vehicle) image samples.
2. **Feature Extraction**: For each image, extract HOG features.
3. **Training**: Use the extracted HOG features to train a Support Vector Machine (SVM) classifier to distinguish between vehicles and non-vehicles.

Once the classifier is trained, it can be saved and loaded for later detection tasks.

---

## Detection Pipeline

During the detection phase, the trained SVM classifier is used to detect vehicles in new images or video frames:

1. **Load the Trained Model**: Load the pre-trained SVM model.
2. **Sliding Window Search**: The sliding window technique is used to search for vehicles across different regions and scales in the image.
3. **Feature Extraction**: For each window, extract HOG features and pass them to the SVM classifier.
4. **Heatmap Creation**: A heatmap is created to visualize the locations of detected vehicles.
5. **Thresholding**: A threshold is applied to the heatmap to remove false positives and refine the detection.
6. **Bounding Boxes**: Bounding boxes are drawn around the detected vehicles for visualization.

---

## Results

The system has been tested on a dataset of vehicles and non-vehicles, achieving high accuracy in detecting vehicles in both images and video frames. The system is robust and can detect vehicles in varying conditions, such as different lighting, angles, and backgrounds.

---

## Usage

### Training Phase (`train_model.py`)

1. Prepare a dataset of vehicle and non-vehicle images.
2. Extract HOG features from these images.
3. Train an SVM classifier using the features.
4. Save the trained model for later use.

```bash
python train_model.py
