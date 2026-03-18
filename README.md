# Computer-Vision-Pipeline-CNN-Image-Classification
### By Javier Fernández Ramos, María Ángeles Muñoz Juan-Dalac and Marta González Pérez
End-to-end Computer Vision pipeline featuring a custom Residual CNN with SiLU activations. Highlighting advanced data curation (pHash), data augmentation, and iterative model optimization for multi-class image recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow/PyTorch](https://img.shields.io/badge/Framework-Deep--Learning-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
This project implements a robust **Convolutional Neural Network (CNN)** pipeline designed to classify images across 10 distinct flower categories. The core focus was to evolve a standard CNN into a **Residual-based architecture**, significantly improving feature extraction and gradient flow.

The project highlights the full machine learning lifecycle: from complex data preprocessing and **leakage prevention** to iterative architectural refinement.

## Key Architectural Features
The final model outperformed baseline versions by implementing:
- **Shortcut Connections (Residual Blocks):** Inline implementation where the output of the first convolution is stored and added to the second, mitigating the vanishing gradient problem.
- **Modern Activation Functions:** Transitioned from ReLU to **SiLU (Sigmoid Linear Unit)** for smoother gradients and improved convergence.
- **Robust Normalization:** Integrated **Batch Normalization** and **Dropout (0.5)** in each block to ensure stability and reduce overfitting.
- **Progressive Abstraction:** Three residual blocks with increasing filter sizes (64, 128, 256).

## Data Engineering and Integrity 
- **Leakage Prevention:** Used **Perceptual Hashing (pHash)** to identify and group near-duplicate images, ensuring they didn't split across training and test sets.
- **Custom Preprocessing:** Developed a standardized resizing logic (336x267) that preserves aspect ratios while ensuring uniform input dimensions.
- **Advanced Augmentation:** Implemented a transformation suite (flips, rotations, zooms, and contrast adjustments) to compensate for the limited dataset size (733 images).

## Performance Results
The architectural shift to shortcut connections led to a significant performance leap:
- **Accuracy:** Improved from **54% to 62.8%**.
- **F1 Score:** 0.628.
- **AUC-ROC:** Achieved almost perfect scores for specific classes (e.g., Tulips, Daisies).
- **Stability:** The ROC curves show early stabilization and consistent discriminative power across all classes.

- **Technical Insight:** An initial discrepancy between high ROC and low accuracy led to the discovery of a label-alignment bug during data import. Fixing this data-level issue was critical to the project's success, proving that model improvements cannot compensate for corrupted data.

## Tech Stack
- Languages: Python
- Libraries: TensorFlow/Keras (or PyTorch), NumPy, Pandas, Matplotlib, OpenCV.
- Techniques: Residual Connections, SiLU Activation, Perceptual Hashing, Data Augmentation.
