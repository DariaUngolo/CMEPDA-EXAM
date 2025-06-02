
# 🧠 CMEPDA-EXAM

**Brain MRI Classification Pipeline for Alzheimer’s Disease Detection**

This project focuses on the development and implementation of a **binary classifier** aimed at distinguishing between subjects diagnosed with **Alzheimer’s Disease (AD)** and **healthy control subjects (CTRL)**. The dataset consists of brain MRI scans from a total of **333 subjects**, including **144 patients with AD** and **189 healthy controls**.

The available data includes **3D brain MRI images in NIfTI format**, alongside **two different brain atlases** used to parcellate the brain into anatomically meaningful regions known as **Regions of Interest (ROIs)**. These atlases segment the brain into **56** and **246 ROIs**, respectively, providing different levels of spatial resolution.

### Feature Extraction and Pipeline Overview

Feature extraction is performed using **MATLAB**, which processes the MRI scans and atlas segmentations to compute descriptive statistics—such as mean intensity, standard deviation, and region volume—for each ROI. These features form the input data for the classification pipeline implemented in **Python**.

### Classification Approaches

The binary classification task is tackled using two complementary approaches:

1. **Classical Machine Learning**, including:
   - **Random Forest (RF)** classifier, with three variants:
     - Standard Random Forest
     - Random Forest combined with **Principal Component Analysis (PCA)** for dimensionality reduction
     - Random Forest with **Recursive Feature Elimination (RFE)** for automated feature selection
   - **Support Vector Machine (SVM)** with customizable kernels (e.g., linear, RBF)

2. **Deep Learning**, implemented through a **3D Convolutional Neural Network (CNN)**, which learns hierarchical features directly from the MRI volumes.

### Evaluation Methodology

For each classical ML classifier configuration, the model is trained and evaluated over **10 independent runs**, using a robust **20-fold cross-validation** strategy to ensure statistical reliability and generalizability. Performance metrics are averaged across runs and folds, and include:

- Accuracy
- Precision
- Recall
- F-1 score
- Area Under the ROC Curve (AUC)

Results are visualized through:

- **ROC curves** summarizing classifier discrimination ability
- **Bar charts** illustrating mean performance metrics along with their confidence intervals

In the case of the **RFE-based Random Forest**, an additional **pie chart** is generated to display the **top 8 most relevant ROIs** contributing to the classification decision, along with their relative feature importances.

### Advanced ROI-based Analysis

The ROIs identified as most informative by RFE are used in the second phase of the project to refine image processing. Specifically, these top-ranked ROIs define a **bounding box** around the brain, which is then used to **crop the MRI volumes** to focus on the most diagnostically relevant areas. This localized cropping facilitates further analysis and potentially improves the deep learning model’s ability to focus on pathological patterns linked to Alzheimer’s Disease.

[CONTINUARE CON LA CNN]

### Execution Modes: Training vs. Inference

The user can choose to run the pipeline in either **training mode** or **inference mode**, depending on the task:

**Training Mode**

- Executes the full pipeline described above:
  - Feature extraction (via MATLAB)
  - Model training using the selected classifier and parameters
  - Performance evaluation with cross-validation and plotting of results
- At the end of training, the entire trained pipeline (including preprocessing and classifier) is saved as a `.joblib` file.
- After training completes, the user is prompted whether they want to classify **single MRI images** extracted from independent datasets, using the newly trained model.

**Inference Mode**

- Requires the user to provide the path to a **pre-trained model file** (`.joblib`) saved in a previous run.
- The pipeline loads the saved model and uses it directly to classify **new individual MRI images** without retraining.
- This mode is optimized for applying the classifier on unseen data efficiently.

---
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

## Pipeline guide

### 1. Feature Extraction (MATLAB)

The MATLAB script `feature_extractor.m` computes the following for each **ROI** defined in the atlas:

- *Mean intensity*

- *Standard deviation*

- *Region volume*

The results are saved to .csv files and read by the Python module for classification.

### 2. Classification (Python)
The Python module performs classification using the following approaches:

- ✅ Random Forest ( --classifier rf )

     - With or without *PCA*

     - With *RFECV* (automated feature selection)

- ✅ SVM ( --classifier svm )

     - With custom parameters (e.g., *RBF kernel*)

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
This project is intended solely for **academic and research purposes**.

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

  
