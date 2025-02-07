# PneumoVision: Pneumonia Classification Using Deep Learning

Developers: Sahiel Bose & Shanay Gaitonde

Research Article: https://medium.com/nextgen-innovations/enhancing-pneumonia-identification-using-vgg16-and-deep-learning-techniques-44ea3bd8e10f

PneumoVision is a deep learning-based system designed to classify chest X-ray images as either Pneumonia or Normal. Using transfer learning and data preprocessing techniques, the model achieves high accuracy with efficient training times. The project is built with TensorFlow/Keras and leverages state-of-the-art models to streamline medical image analysis.

Project Highlights
Preprocessing: Extensive data augmentation techniques are applied to make the model more robust.
Model: Pretrained VGG16 is used as the backbone, with a custom dense layer classifier.
Callbacks: Adaptive learning with ReduceLROnPlateau and EarlyStopping ensures efficient training.
Class Weighting: Addresses imbalanced datasets for improved model fairness.
Scalability: Handles chest X-ray images with customizable resolutions.
Model Description
1. Preprocessing Pipeline
Normalization: Pixel values are scaled to the range [0, 1] using rescale=1./255.
Data Augmentation: Enhances generalization by randomly applying:
Rotation: Up to ±20°
Shifts: Horizontal and vertical translations by up to 20%.
Shear Transformations: Up to 20%.
Zoom: Randomly zooms images by up to 20%.
Horizontal Flip: Adds robustness to positional variation.
Brightness Adjustments: Varies brightness from 80% to 120%.
Validation Split: Automatically splits the training data into 80% training and 20% validation sets.
2. Model Architecture
Base Network: The pretrained VGG16 architecture is used with weights initialized from ImageNet.
The convolutional layers are frozen during initial training to preserve learned feature extraction capabilities.
Custom Classifier:
A Flatten layer converts the feature maps into a 1D vector.
A Dense Layer (256 units, ReLU) adds learnable parameters for classification.
A Dropout Layer (50%) prevents overfitting.
A Dense Layer (1 unit, Sigmoid) performs binary classification (Pneumonia vs. Normal).
Why VGG16? The convolutional blocks of VGG16 are excellent feature extractors, especially for image classification tasks. By fine-tuning the last block, we adapt these features specifically for the chest X-ray dataset.

3. Training Strategy
Optimizer: Adam optimizer with an initial learning rate of 1e-4.
Loss Function: Binary Cross-Entropy for classification.
Class Weights: Automatically computed to counter class imbalance.
Callbacks:
ReduceLROnPlateau: Reduces learning rate by a factor of 0.5 if validation loss plateaus for 3 epochs.
EarlyStopping: Stops training early if validation loss does not improve for 5 consecutive epochs.
Training Setup:

Batch Size: 32
Target Image Size: Adjustable to either 128×128 or 150×150 pixels.
Epochs: 20 (adaptive based on callbacks).
4. Post-Training Evaluation
Evaluate the model's performance on a test set.
Metrics:
Accuracy
Loss
Confusion matrix visualization (optional for deeper insights).
Dataset
The dataset includes chest X-ray images split into the following:

Train Directory: Used for training the model with augmentation applied.
Validation Split: Automatically extracted from the training set.
Test Directory: Evaluated post-training to test the model's generalizability.
The images are categorized into:

NORMAL
PNEUMONIA
Results
Performance Metrics:

Training and validation accuracy typically exceed 90% after fine-tuning.
Loss decreases consistently due to robust preprocessing and adaptive callbacks.
Visualization:

Accuracy and loss graphs during training show convergence.

Under Dyne Research

