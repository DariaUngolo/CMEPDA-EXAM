# 🧠 CMEPDA-EXAM

**Brain MRI Classification Pipeline for Alzheimer’s Disease Detection**

This repository implements an automatic pipeline for classifying brain MRI images (NIfTI format) between subjects with **Alzheimer’s Disease (AD)** and **healthy controls (CTRL)**. Feature extraction is performed in MATLAB, while classification is done in Python using various machine learning techniques.

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

---

## 🚀 How to Run

Execute the `main.py` script with the following arguments from the terminal or PowerShell:
``` bash
python main.py \
  --folder_path "path/to/nifti_folder" \
  --atlas_file "path/to/original_atlas.nii.gz" \
  --atlas_file_resized "path/to/resampled_atlas.nii.gz" \
  --atlas_txt "path/to/atlas_labels.txt" \
  --metadata_csv "path/to/metadata.csv" \
  --output_prefix "path/to/output_prefix" \
  --matlab_path "path/to/MATLAB_folder" \
  --classifier {rf, svm} \
  --n_iter N_ITER \
  --cv CV \
  --kernel {linear, rbf}
```

---

## Steps

### 1. Feature Extraction (MATLAB)

The MATLAB script feature_extractor.m computes the following for each ROI defined in the atlas:

- Mean intensity

- Standard deviation

- Region volume

The results are saved to .csv files and read by the Python module for classification.

### 2. Classification (Python)
The Python module performs classification using the following approaches:

- ✅ Random Forest (--classifier rf)

     - With or without PCA

     - With RFECV (automated feature selection)

- ✅ SVM (--classifier svm)

     - With custom parameters (e.g., RBF kernel)

### 3. Performance Metrics:
- Accuracy

- Precision

- Recall

- AUC (Area Under the Curve)

- Confidence intervals (binomial, bootstrap-based)

  
### 4. Output
At the end of the execution, the following are generated:

- CSV files with the extracted features

- Predicted class for each subject

- Plots:

  1. ROC curve

  2. Performance bar chart

  3. Feature importance (only for Random Forest with RCEFV)

- Logs and saved trained models

---

## Documentation
INSERIRE LINK DOCUMENTAZIONE

--- 

## 📄 License
This project is intended solely for academic and research purposes.

---

## 👤 Authors

- **Brando Spinelli**
- **Daria Ungolo**

---

## 💬 Contact
Have questions or issues? Open an issue or contact:
- 📧 [b.spinelli2@studenti.unipi.it]
- 📧 [d.ungolo@studenti.unipi.it]

---

## ⭐ Contributing

If you'd like to contribute:

 1. Fork the repository

 2. Create a new branch (git checkout -b new-feature)

 3. Commit your changes (git commit -am 'Add new feature')

 4. Push to GitHub (git push origin new-feature)

 5. Open a pull request!

  
