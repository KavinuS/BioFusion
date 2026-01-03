# BioFusion - Gastric Cancer Histopathology Classification

A comprehensive deep learning project for classifying gastric cancer histopathology tissue images using Vision Transformers (ViT) and other advanced models.

## 🎯 Project Overview

This repository contains code for training and evaluating deep learning models on the Gastric Cancer Histopathology Tissue Image Dataset (GCHTID), which includes 8 tissue classes:
- **ADI**: Adipose
- **DEB**: Debris  
- **LYM**: Lymphocytes
- **MUC**: Mucus
- **MUS**: Smooth Muscle
- **NOR**: Normal Colon Mucosa
- **STR**: Cancer-associated Stroma
- **TUM**: Tumor

## 🚀 Features

### Vision Transformer (ViT) Model
- **Stain Normalization**: Macenko method for H&E histopathology images
- **Weighted Focal Loss**: Addresses class imbalance and hard examples
- **Class-Specific Weighting**: Penalizes critical confusion pairs
- **Two-Phase Training**: 10 epochs classifier training + 10 epochs fine-tuning
- **GPU Support**: Automatic CUDA detection and usage

### Multiple Model Architectures
- ViT-Base (Vision Transformer)
- ResNet50
- DenseNet121
- EfficientNet-B4
- CTransPath (Histopathology-specific)
- Hybrid Ensemble Models

## 📁 Repository Structure

```
BioFusion/
├── VIT model.ipynb              # Main ViT training notebook (local)
├── VIT_model_colab.py          # Colab version with Kaggle integration
├── model_uni.py                # Original ViT/UNI model script
├── shared_utilities.py         # Shared functions for all models
├── improvements_implementation.py  # Stain norm, focal loss, class weighting
├── model_*.py                  # Other model implementations
├── COLAB_SETUP_INSTRUCTIONS.md # Guide for Google Colab setup
├── Dataset/                    # Dataset folder (excluded from git - too large)
└── Documentation files         # Various guides and summaries
```

## 🛠️ Setup

### Local Environment

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install timm transformers torchstain scikit-learn matplotlib seaborn
   ```

2. **Dataset Setup**:
   - Place dataset in `Dataset/HMU-GC-HE-30K/all_image/`
   - Dataset structure: `all_image/{CLASS_NAME}/*.png`

3. **Run Training**:
   - Open `VIT model.ipynb`
   - Run cells sequentially
   - GPU will be automatically detected and used

### Google Colab

1. **Upload Files**:
   - Upload `shared_utilities.py` to Colab
   - Upload `kaggle.json` when prompted

2. **Run Cells**:
   - Copy cells from `VIT_model_colab.py` into Colab notebook
   - Run sequentially
   - Models automatically save to Google Drive

See [COLAB_SETUP_INSTRUCTIONS.md](COLAB_SETUP_INSTRUCTIONS.md) for detailed setup.

## 📊 Model Files

**Note**: Model files (`.pt`) are excluded from this repository due to GitHub's 100MB file size limit. 

Trained models can be:
- **Saved locally**: `vit_final.pt`, `vit_results.json`
- **Uploaded to Google Drive**: Automatic in Colab, manual for local training
- **Shared with judges**: Upload to Google Drive and share folder

## 🎓 Training Configuration

- **Epochs**: 20 total (10 Phase 1 + 10 Phase 2)
- **Batch Size**: 32
- **Learning Rate**: 
  - Phase 1: 1e-3 (classifier head)
  - Phase 2: 1e-4 (head), 1e-5 (backbone)
- **Loss Function**: Weighted Focal Loss (alpha=0.25, gamma=2.0)
- **Early Stopping**: Patience of 5 epochs

## 📈 Improvements Implemented

1. **Stain Normalization**: Macenko method for consistent H&E staining
2. **Focal Loss**: Better handling of hard examples and class imbalance
3. **Class Weighting**: Increased penalty for critical confusion pairs:
   - NOR ↔ TUM (Normal vs Tumor)
   - MUC ↔ ADI (Mucus vs Adipose)
   - DEB ↔ STR (Debris vs Stroma)

## 📝 Key Files

- **VIT model.ipynb**: Complete training notebook for local execution
- **VIT_model_colab.py**: Colab-ready version with Kaggle dataset integration
- **shared_utilities.py**: Common functions (data loading, training, evaluation)
- **improvements_implementation.py**: Stain norm, focal loss, class weighting

## 🔗 Dataset

Dataset: [Gastric Cancer Histopathology Tissue Image Dataset (GCHTID)](https://www.kaggle.com/datasets/orvile/gastric-cancer-histopathology-tissue-image-dataset)

- **Source**: Kaggle
- **Size**: ~31,000 images
- **Classes**: 8 tissue types
- **Image Size**: 224x224 pixels

## 📚 Documentation

- `COLAB_SETUP_INSTRUCTIONS.md`: Google Colab setup guide
- `FINAL_MODELS_GUIDE.md`: Model comparison and usage
- `IMPROVEMENT_ANALYSIS.md`: Detailed improvement explanations
- `MODEL_COMPARISON_ANALYSIS.md`: Performance comparisons

## 🤝 Contributing

This is a research project for gastric cancer histopathology classification. For questions or issues, please open an issue on GitHub.

## 📄 License

This project is for research purposes. Dataset is released under Creative Commons Attribution 4.0 International License (CC BY 4.0).

## 🙏 Acknowledgments

- Dataset creators: Shenghan Lou et al.
- Model architectures: timm, transformers libraries
- Stain normalization: torchstain library

---

**Repository**: [https://github.com/KavinuS/BioFusion](https://github.com/KavinuS/BioFusion)

