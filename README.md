# DeepLearning 2025-26 - Aniket Tiwari

A comprehensive deep learning repository focusing on neural networks, CNN architectures, and practical implementations using TensorFlow and Keras.

## 📁 Repository Overview

This repository contains deep learning projects and implementations, with a primary focus on computer vision tasks using Convolutional Neural Networks (CNNs).

## 🚀 Projects

### 1. MNIST Digit Recognition using CNN
A complete implementation of a Convolutional Neural Network for recognizing handwritten digits from the MNIST dataset.

**Features:**
- Custom CNN architecture with multiple convolutional and pooling layers
- Data preprocessing and normalization
- Model training with validation splits
- Comprehensive evaluation metrics
- Visualization of results and performance metrics

**Model Architecture:**
- Conv2D (32 filters) + MaxPooling + Dropout
- Conv2D (64 filters) + MaxPooling + Dropout
- Flatten + Dense (128 units) + Dropout
- Output Layer (10 classes with softmax)

**Performance:**
- Test Accuracy: **99.30%**
- Training conducted over 10 epochs
- Uses Adam optimizer with sparse categorical crossentropy loss

### 2. Dataset Loading Examples
Demonstrates multiple methods for loading datasets in Google Colab:
- Direct file upload to Colab
- Google Drive mounting
- Kaggle API integration using tokens

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow & Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Evaluation metrics
- **Kaggle API** - Dataset downloads

## 📊 Datasets

- **MNIST Dataset**:  Handwritten digit recognition (60,000 training images, 10,000 test images)
- **Tomato Disease Dataset**: Plant disease classification (via Kaggle)

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub
```

### Clone the Repository
```bash
git clone https://github.com/AniketTiwari13/DeepLearning_2025-26-Aniket-Tiwari.git
cd DeepLearning_2025-26-Aniket-Tiwari
```

## 💻 Usage

### Running the MNIST CNN Model
```python
# Load the required libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Train the model (see notebooks for complete implementation)
```

### Loading Datasets from Kaggle
```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("dataset-name")
print("Path to dataset files:", path)
```

## 📈 Results & Visualizations

The notebooks include comprehensive visualizations:
- Pixel intensity distributions
- Sample images from datasets
- Class distribution plots
- Training vs.  validation accuracy/loss curves
- Confusion matrices
- Classification reports

## 📂 File Structure

```
DeepLearning_2025-26-Aniket-Tiwari/
│
├── DeepLearning. ipynb                              # Dataset loading examples
├── mnist_digit_recognition_cnn_deep_learning.ipynb # Complete CNN implementation
└── README.md                                        # Project documentation
```

## 🎯 Key Learnings

- Implementing CNN architectures for image classification
- Data preprocessing and normalization techniques
- Using dropout layers to prevent overfitting
- Evaluating model performance with multiple metrics
- Visualizing training progress and results
- Working with different data sources (Kaggle, Google Drive, local files)

## 🔍 Model Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive predictions per class
- **Recall**:  Ability to find all positive instances
- **F1-Score**:  Harmonic mean of precision and recall
- **Confusion Matrix**:  Detailed breakdown of predictions vs. actual labels

## 📝 Future Work

- [ ] Implement advanced CNN architectures (ResNet, VGG, etc.)
- [ ] Explore transfer learning techniques
- [ ] Add more computer vision projects
- [ ] Implement data augmentation strategies
- [ ] Deploy models using TensorFlow Serving or Flask
- [ ] Add RNN/LSTM implementations for sequence data

## 👨‍💻 Author

**Aniket Tiwari**
- GitHub: [@AniketTiwari13](https://github.com/AniketTiwari13)

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  Feel free to check the issues page if you want to contribute.

## 🙏 Acknowledgments

- MNIST Dataset:  Yann LeCun and Corinna Cortes
- TensorFlow and Keras teams for excellent documentation
- Kaggle community for datasets and inspiration

---

**Note**: This repository is part of a Deep Learning course for the academic year 2025-26.
