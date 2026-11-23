# CNN Handwritten Digit Recognition

A Deep Learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project includes a Streamlit web application for image upload, digit prediction, prediction history, and model performance visualization.

## Project Overview

Handwritten Digit Recognition is an image classification application that identifies numerical digits from handwritten images.

This project uses a CNN model built with TensorFlow and Keras. The trained model processes input images and predicts the corresponding digit with a confidence score.

## Objectives

- Build a CNN model for handwritten digit classification.
- Train and evaluate the model using the MNIST dataset.
- Perform image preprocessing for better prediction.
- Develop an interactive Streamlit application.
- Display predictions and model performance visualizations.

## Dataset

**Dataset:** MNIST Handwritten Digit Dataset

The MNIST dataset contains:
- 60,000 training images.
- 10,000 testing images.
- 28 × 28 pixel grayscale images.
- 10 digit classes (0–9).

## CNN Model Architecture

The project uses a Convolutional Neural Network with layers for feature extraction and classification.

The architecture includes:
- Convolutional layers
- Max Pooling layers
- Flatten layer
- Fully Connected (Dense) layers
- Dropout layer
- Softmax output layer

The model predicts one of the ten digit classes.

## Model Performance

- Dataset: MNIST
- Training epochs: 10
- Test accuracy: 99.06%

*Note: The reported accuracy is on the MNIST test dataset. Performance on independently uploaded handwritten images may vary.*

## Streamlit Application Features

- Manual image upload.
- Camera image capture.
- Automatic image loading from the input_images folder.
- Single-digit prediction.
- Multiple-digit image processing.
- 5 × 2 grid-based digit detection.
- Prediction confidence display.
- Prediction history.
- Digit frequency analytics.
- Model performance dashboard.
- Confusion matrix and training graphs.

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

## Project Structure

```text
CNN_Image_Classification_Project/
│
├── app.py
├── .gitignore
├── requirements.txt
│
├── models/
│   └── mnist_cnn_model.keras
│
├── input_images/
│   ├── digit2.png
│   ├── digit4.png
│   └── digit7.png
│
└── development/
    ├── cnn_model_training.py
    ├── cnn_model_evaluation.py
    ├── cnn_model_summary.txt
    ├── mnist_preprocessing.py
    ├── training_history.npz
    ├── evaluation_results.npz
    ├── training_accuracy.png
    ├── training_loss.png
    ├── confusion_matrix.png
    └── actual_vs_predicted.png
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Bhavana-hm/CNN-Handwritten-Digit-Recognition.git
```

### 2. Navigate to the project folder

```bash
cd CNN-Handwritten-Digit-Recognition
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## Future Enhancements

- Improve recognition of different handwriting styles.
- Support more complex handwritten inputs.
- Improve multi-digit segmentation.
- Deploy the application online.

## Conclusion

This project demonstrates the application of Convolutional Neural Networks in handwritten digit recognition. It combines deep learning with an interactive Streamlit interface to provide digit predictions and model performance analysis.

## Author

**Bhavana H M**

**Batch:** B

**Project:** CNN Handwritten Digit Recognition

**Dataset:** MNIST
