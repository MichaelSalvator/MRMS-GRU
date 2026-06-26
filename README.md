# MRMS-GRU:A Unified Spatio-Temporal Framework Enables Robust Monitoring of The Changes in Mangroves in The Vast Tropical Coastal Areas of Southern China

This project is a pixel-level classification and change detection framework for mangroves based on time-series multispectral remote sensing imagery (e.g., Sentinel-2). The repository includes traditional machine learning baseline models (Random Forest, XGBoost, SVM) and proposes a custom deep learning architecture named **MRMS-GRU**, which integrates multi-scale 1D Convolutional Neural Networks (1D-CNN) and Bidirectional Gated Recurrent Units (BiGRU).

The framework supports ablation studies across different time-series lengths (Full, Half, Quarter, Eighth) and incorporates Focal Loss optimization to address class imbalance issues.

## Directory Structure & Features

* **Data Processing Module**
  * `Data_process.py`: Reads raw remote sensing images (using `rasterio`), performs spatial cropping based on specified coordinates, extracts band data, and saves them in `.npy` format. It also generates histogram-equalized RGB visualization preview images.
  * `Data_process2.py`: Merges all independent temporal `.npy` image data along the time dimension to generate a globally unified time-series tensor `merged_data.npy`.
  * `Data_process3.py`: Automatically samples positive (mangrove changes), negative (no changes), and background pixels based on manually annotated binary masks (`mask_change.png`, `mask_nochange.png`). It supports truncating time-series data into different lengths (1, 1/2, 1/4, 1/8) for robustness testing and generates the final training and validation sample sets.
* **Model and Data Loading Module**
  * `Dataloader.py`: Defines the `RemoteSensingDataset` class to convert `.npy` samples into PyTorch-compatible Datasets and DataLoaders.
  * `model3.py`: The core deep learning model script that defines the **MRMS-GRU** network. It includes multi-scale convolutional feature extractors, residual connections, and sequence modeling layers.
* **Model Training Module**
  * `train_DL_2.py`: Deep learning model training script. It utilizes the AdamW optimizer and Focal Loss, and automatically calculates and saves metrics like Accuracy, Precision, Recall, F1-Score, and Cohen's Kappa coefficient into the `logs/` directory.
  * `train_ML_2.py`: Traditional machine learning model training script (includes RandomForest, XGBoost, SVM, etc.).
* **Prediction and Inference Module**
  * `predict_DL_2.py`: Loads the trained deep learning model weights, performs sliding inference over the entire image, and outputs category predictions as a colored mask image (Black: Background, Teal: Unchanged, Red: Changed).
  * `predict_ML.py`: Loads the serialized traditional machine learning model (`.pth` via pickle), performs full-image inference, and outputs a colored classification map.

## Installation & Requirements

Please ensure your computer is running a Python 3.8+ environment. A CUDA-enabled GPU is highly recommended for deep learning training.

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
