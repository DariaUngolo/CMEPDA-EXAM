# 🧠 CMEPDA-EXAM

**Brain MRI Classification Pipeline for Alzheimer’s Disease Detection**

This repository implements an automatic pipeline for classifying brain MRI images (NIfTI format) between subjects with **Alzheimer’s Disease (AD)** and **healthy controls (CTRL)**. Feature extraction is performed in MATLAB, while classification is done in Python using various machine learning techniques.

---

## 📁 Project Structure



CMEPDA-EXAM/
├── ML_main/
│ └── main.py # Script principale da eseguire
├── ML_codes/
│ ├── classifiers_unified.py # Classificatori ML (SVM, RF, ecc.)
│ ├── performance_scores.py # Funzioni per il calcolo delle metriche
│ └── feature_extractor.py # Interfaccia Python-MATLAB
├── MATLAB/
│ └── feature_extractor.m # Script MATLAB per l’estrazione delle feature
├── data/ # Cartella con file NIfTI, atlante e metadata
└── README.md # Questo file


---

## ⚙️ Requirements

### 🐍 Python

- Python 3.9
- Key packages:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
  - `matlab.engine` (MATLAB-Python interface)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### 🧮 MATLAB
- MATLAB R2024b (or compatible)

- MNI152 brain atlas (in .nii.gz format)



## 🚀 How to Run

Execute the main.py script with the following arguments from the terminal or PowerShell:

