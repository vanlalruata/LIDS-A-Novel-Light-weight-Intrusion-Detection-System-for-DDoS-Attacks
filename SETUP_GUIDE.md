# LIDS - Lightweight Intrusion Detection System
## Setup and Usage Guide

## Prerequisites
- Python 3.7 or higher
- CUDA-capable GPU (optional, but recommended for training)

## Installation

### 1. Install Dependencies
```bash
cd LIDS
pip install -r requirements.txt
```

### 2. Dataset Setup
Place your dataset files in the appropriate directory:
- **CIC-DDoS2019**: `H:/Datasets/CIC-DDoS2019/01-12/` (or modify PATH in main.py)
- **BoT-IoT**: `H:/Datasets/BoT-IoT/`
- **TON_IoT**: `H:/Datasets/TON_IoT/Processed_datasets/Processed_Network_dataset/`

## Usage

### Running the Main Program
```bash
cd LIDS
python main.py
```

### Menu Options

**Option 1: Create Processed Dataset**
- Loads raw dataset files
- Performs preprocessing (removes NaN, duplicates, constant features)
- Applies scaling and feature selection
- Saves processed dataset to `Datasets/Processed_<dataset>.csv`

**Option 2: Create PCA Dataset**
- Loads processed dataset
- Applies PCA dimensionality reduction (41 components by default)
- Splits into train/test sets
- Saves to `Datasets/train_PCA_<dataset>.csv` and `Datasets/test_PCA_<dataset>.csv`

**Option 3: Visualization**
- Generates various plots:
  - File size distribution after preprocessing
  - Class distribution (binary and multiclass)
  - PCA variance analysis
  - 3D PCA visualization

**Option 4: Binary Classification Training**
- Trains LCNN model for binary classification (Normal vs Attack)
- Default: 100 epochs, batch size 1024
- Saves model to `PretrainedModel/modelv<N>.pth`
- Saves metrics to `Datasets/train_val_metrics_<dataset>.csv`

**Option 5: Multiclass Classification Training**
- Trains LCNN model for multiclass classification (13 classes)
- Default: 100 epochs, batch size 1024
- Saves model to `PretrainedModelMulti/modelMultiv<N>.pth`
- Saves metrics to `Datasets/train_val_metrics_multi_<dataset>.csv`

**Option 6: Plot Training Results**
- Generates accuracy and loss plots from training metrics

## Project Structure
```
LIDS/
├── main.py                 # Entry point
├── data_utils.py          # Basic data utilities
├── eval_tools.py          # Evaluation metrics
├── vis_utils.py           # Visualization utilities
├── requirements.txt       # Python dependencies
├── Proposed/
│   ├── data_utils.py      # Advanced preprocessing
│   ├── data_loader.py     # PyTorch data loaders
│   ├── model.py           # LCNN model definitions
│   ├── train.py           # Training logic
│   └── pca_analysis.py    # PCA analysis tools
├── Datasets/              # Processed datasets (created at runtime)
├── Images/                # Generated visualizations
├── PretrainedModel/       # Binary classification models
└── PretrainedModelMulti/  # Multiclass models
```

## Typical Workflow

1. **First Time Setup**:
   ```bash
   python main.py
   # Select option 1 to create processed dataset
   # Select option 2 to create PCA dataset
   ```

2. **Training**:
   ```bash
   python main.py
   # Select option 4 for binary classification
   # OR option 5 for multiclass classification
   ```

3. **Visualization**:
   ```bash
   python main.py
   # Select option 3 for dataset visualizations
   # Select option 6 for training plots
   ```

## Configuration

Edit [`main.py`](main.py:3) to modify:
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Batch size for training (default: 1024)
- `PATH`: Dataset directory path
- `n_rows`: Number of rows to load per file (default: 500000)

## Troubleshooting

### Import Errors
Ensure you're running from the LIDS directory:
```bash
cd LIDS
python main.py
```

### Missing Dataset Files
Update the `PATH` variable in [`main.py`](main.py:14) to point to your dataset location.

### CUDA Out of Memory
Reduce `BATCH_SIZE` in [`main.py`](main.py:13) or use CPU by modifying device selection in [`train.py`](Proposed/train.py:25).

### Missing Dependencies
Install all requirements:
```bash
pip install -r requirements.txt
```

## Model Architecture

### Binary Classification (LCNNModel)
- Input: 10 PCA components
- Conv1D layer (1→16 channels)
- MaxPool2D
- Fully connected layers (48→64→2)
- Dropout (0.6)

### Multiclass Classification (LCNNModelMulti)
- Input: 10 PCA components
- Conv1D layers (1→32→64 channels)
- MaxPool2D
- Fully connected layers (224→128→13)
- Dropout (0.5)

## Performance Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Results are displayed during testing and saved to CSV files.

## Notes

- The system automatically creates necessary directories
- Models are versioned (modelv1, modelv2, etc.)
- Dataset prefix is tracked for multi-dataset support
- GPU acceleration is used automatically if available