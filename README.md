# Face Mask Detection System

## Overview
This project implements a real-time face mask detection system using computer vision and deep learning. It leverages Convolutional Neural Networks (CNNs) to classify whether individuals are wearing face masks, utilizing webcam input for real-time detection.

## Objective
- Develop a system to detect face mask usage in real-time using computer vision and deep learning techniques.

## Features
- **Real-Time Detection**: Utilizes OpenCV to process webcam feed for live mask detection.
- **CNN Model**: Employs a Convolutional Neural Network for accurate image classification (Mask/No Mask).
- **Data Preprocessing**: Includes grayscale conversion, histogram equalization, and data augmentation.
- **Model Training**: Trains the model on labeled images stored in the `images/` directory.
- **Output**: Displays detection results with bounding boxes and labels on the video feed.

## Model Architecture
The CNN model is structured as follows:
- **Input Layer**: Accepts 32x32x1 grayscale images.
- **Convolutional Layers**:
  - Two layers with 32 filters (3x3), ReLU activation.
  - Two layers with 64 filters (3x3), ReLU activation.
- **Pooling Layers**: MaxPooling (2x2) after each convolutional block to reduce spatial dimensions.
- **Dropout**: Applied with a 0.5 rate to prevent overfitting.
- **Fully Connected Layers**:
  - Flatten layer to convert 2D features to 1D.
  - Dense layer with 64 units, ReLU activation.
  - Output layer with softmax activation for binary classification (Mask/No Mask).

## Requirements
To run this project, install the following Python packages:
```bash
pip install opencv-python numpy keras tensorflow scikit-learn matplotlib
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `images/` directory contains labeled data in subfolders `0` (Mask) and `1` (No Mask).

## Usage
### Training the Model
1. Run the training script to generate the model file `MyTrainingModel.h5`:
   ```bash
   python train.py
   ```
2. The script expects labeled images in the `images/` directory with subfolders `0` and `1`.

### Real-Time Detection
1. Connect a webcam to your device.
2. Run the detection script:
   ```bash
   python detect.py
   ```
3. Press `q` to exit the detection window.

### Data Collection
- The script includes functionality to capture and save face images without masks to the `images/face_without_mask/` directory for further training.
- Run the data collection script:
   ```bash
   python collect_data.py
   ```
- The script saves up to 500 face images and draws bounding boxes on detected faces.

## File Structure
- `train.py`: Script for training the CNN model.
- `detect.py`: Script for real-time face mask detection using a webcam.
- `collect_data.py`: Script for collecting face images without masks.
- `images/`: Directory containing labeled training data (subfolders `0` and `1`).
- `MyTrainingModel.h5`: Trained model file.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade for face detection.

## Dependencies
- **OpenCV**: For image processing and webcam handling ([Documentation](https://docs.opencv.org/)).
- **Keras**: For building and training the CNN model ([Documentation](https://keras.io/)).
- **TensorFlow**: Backend for Keras ([Documentation](https://www.tensorflow.org/)).
- **Matplotlib**: For visualizing data distribution ([Documentation](https://matplotlib.org/)).

