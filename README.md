# LIDS — Lightweight Intrusion Detection System

This repository contains a PCA-driven, CNN-based intrusion detection pipeline with utilities for preprocessing, visualization, and training on multiple datasets:
- CICDDoS2019
- BoT-IoT
- TON_IoT

The project supports both binary and multi-class classification flows. For CICDDoS2019 specifically, a refined CNN architecture and SHAP-based explainability are integrated into the multi-class trainer.

### This project source code has been used for the paper:
DOI : 10.1002/spy2.70148<br/>
Title : LIDS: A Novel Framework for Lightweight Detection of DDoS Attacks<br/>
Journal : Security and Privacy<br/>
Publisher : Wiley<br/>

> If you find the code useful, you are requested to cite the paper.

## Key Features
- Unified CLI workflow via main.py with menu options for end-to-end runs
- Dataset-agnostic preprocessing and PCA pipeline
- CNN models for binary and multi-class tasks
- CICDDoS2019-specific enhancements in multi-class mode:
  - Improved Conv1d → BatchNorm1d → ReLU → MaxPool1d → FC → ReLU → Dropout → FC
  - Class-weighted CrossEntropyLoss
  - AdamW optimizer with early stopping
  - Optional SHAP KernelExplainer to highlight top PCA features per class
- Metrics and artifacts saved to disk for later visualization


## Requirements
- Python 3.8+ (3.9–3.11 recommended)
- Linux/Mac/Windows 10/11
- CUDA-capable GPU optional (training will fall back to CPU if CUDA unavailable)

Install Python packages:

```
cd LIDS
pip install -r requirements.txt
# Optional (for SHAP explainability on CIC):
pip install shap
```

Notes:
- torchviz and graphviz are optional; if you enable model graphing, ensure Graphviz is installed system-wide and on PATH.
- If you see matplotlib headless warnings, the code already uses Agg backend where needed.


## Datasets and Paths
main.py prompts you to choose the dataset and internally maps choices to default paths:
- CICDDoS2019: \Datasets\CIC-DDoS2019\01-12
- BoT-IoT: \Datasets\BoT-IoT
- TON_IoT: \Datasets\TON_IoT\Processed_datasets\Processed_Network_dataset

If your datasets are elsewhere, either:
- Create those directories and place data accordingly, or
- Edit LIDS/main.py to point to your actual dataset locations (lines where dataset_path is set for options 1–3).

The currently active dataset prefix is persisted in:
- LIDS\Datasets\current_dataset_prefix.txt

That prefix is used to name outputs like processed CSVs, metrics, and model files.


## Project Structure (high level)
- LIDS/main.py — Menu-driven entry point
- LIDS/Proposed/
  - data_utils.py — preprocessing, PCA, dataset creation, helpers
  - data_loader.py — Dataset and DataLoader wrappers (binary and multi)
  - model.py — CNN architectures for binary and multi-class (default)
  - train.py — Training loops; CICDDoS2019-specific multi-class branch with SHAP
  - CNN_model.py — Reference CNN source that inspired CIC enhancements
- LIDS/vis_utils.py — Existing plotting utilities for training curves, sizes, etc.
- LIDS/Datasets/ — Processed datasets, PCA outputs, metrics CSVs, and prefix file
- LIDS/Images/ — Visual outputs; SHAP plots go to Images/shap
- LIDS/PretrainedModel, LIDS/PretrainedModelMulti — Saved models


## Quick Start
1) Install dependencies
- See Requirements above.

2) Run the CLI
```
python LIDS/main.py
```

3) Choose from the menu:
- 1. Proposed Preprocessing: Create Dataset
  - Select dataset: 1=CICDDoS2019, 2=BoT-IoT, 3=TON_IoT
  - This sets the current prefix and generates processed datasets
- 2. Create PCA Dataset using processed dataset
  - Generates PCA features and saves train/test PCA CSVs
- 3. Visualization
  - Uses existing vis_utils and PCA analysis to show distributions and PCA variance
- 4. Proposed Model Binary Classification
  - Trains binary CNN (uses LCNNModel)
- 5. Proposed Model Multi Classification
  - Trains multi-class model
  - If current prefix starts with CIC (e.g., CICDDoS2019), it uses the improved CNN + class weights + AdamW + early stopping; after evaluation it attempts SHAP explainability and saves per-class plots
  - For BoT-IoT and TON_IoT, it uses the default multi-class CNN flow without SHAP
- 6. Plot Training and Validation
  - Plots accuracy/loss for binary and multi-class runs using vis_utils


## Outputs and Artifacts
- Processed and PCA datasets: LIDS/Datasets/*.csv
  - train_PCA_<prefix>.csv, test_PCA_<prefix>.csv
- Training curves (CSV):
  - LIDS/Datasets/train_val_metrics_<prefix>.csv (binary)
  - LIDS/Datasets/train_val_metrics_multi_<prefix>.csv (multi)
- Saved models:
  - LIDS/PretrainedModel/<prefix>_modelvX.pth (binary)
  - LIDS/PretrainedModelMulti/<prefix>_modelMultivX.pth (multi)
- SHAP (CIC only, multi-class):
  - LIDS/Images/shap/shap_top_features_class_*.png


## CICDDoS2019-specific Notes
- The trainer detects the dataset by reading LIDS/Datasets/current_dataset_prefix.txt; any prefix beginning with "cic" (case-insensitive) activates the enhanced path.
- The improved CNN stack is integrated inline in Proposed/train.py to minimize code changes elsewhere.
- SHAP uses KernelExplainer on a capped subset of the test set to control runtime and memory (defaults: background ≤200, explained ≤800, nsamples≈200). Adjust these in run_shap_kernel_explainer_per_class if needed.
- If shap is not installed, training proceeds and SHAP is skipped gracefully.


## Troubleshooting
- Dataset paths not found
  - Edit LIDS/main.py to set your dataset directories or mirror the default Windows paths.
- CUDA/GPU not used
  - The code auto-selects CUDA if available; ensure proper PyTorch build and drivers. Otherwise it runs on CPU.
- Graphviz/torchviz errors
  - Either install Graphviz and add it to PATH or ignore model graphing; it’s optional and not required for training.
- SHAP memory/time usage
  - Reduce nsamples, cap_background, or cap_explain in Proposed/train.py’s SHAP helper. Ensure sufficient RAM.
- PCA columns mismatched
  - The loaders select columns named "PC 1", "PC 2", ...; ensure PCA creation step completed for the chosen prefix.
- Class mapping
  - For multi-class, string labels are mapped to numeric ids. Binary mode ensures labels are 0/1.


## Repro Tips
- Keep the same current_dataset_prefix between PCA and training steps; switching datasets requires re-running steps 1 and 2 for that dataset.
- Record the environment:
  - pip freeze > freeze.txt
  - torch.cuda.is_available() to confirm GPU availability
- For consistent SHAP plots, randomness is seeded for sampling test/background sets in the SHAP routine.


## License
This project is provided as-is for research and educational purposes. Please verify dataset licensing and usage terms for CICDDoS2019, BoT-IoT, and TON_IoT independently.

