
# 🧠 CMEPDA-EXAM

**Brain MRI Classification Pipeline for Alzheimer’s Disease Detection**

This project focuses on the development and implementation of a **binary classifier** aimed at distinguishing between subjects diagnosed with **Alzheimer’s Disease (AD)** and **healthy control subjects (CTRL)**. The dataset consists of brain MRI scans from a total of **333 subjects**, including **144 patients with AD** and **189 healthy controls**.

The available data includes **3D brain MRI images in NIfTI format**, alongside **two different brain atlases** used to parcellate the brain into anatomically meaningful regions known as **Regions of Interest (ROIs)**. These atlases segment the brain into **56** and **246 ROIs**, respectively, providing different levels of spatial resolution.

### Feature Extraction and Pipeline Overview

Feature extraction is performed using **MATLAB**, which processes the MRI scans and atlas segmentations to compute descriptive statistics—such as mean intensity, standard deviation, and region volume—for each ROI. These features form the input data for the classification pipeline implemented in **Python**.

To ensure consistency in spatial alignment, the pipeline automatically verifies whether the input atlas has the same resolution and spatial dimensions as the MRI images. If a mismatch is detected, the atlas is **resampled** to match the image geometry. This operation ensures correct anatomical correspondence between regions and voxel intensities. The newly resampled atlas is then saved to disk and used for all subsequent feature extraction steps.

This alignment step is essential for preserving anatomical accuracy and avoiding artifacts in ROI-level statistics. Once the features are extracted, they serve as the input to the classification stage of the pipeline, implemented in Python.

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


### 🧪 Unit Testing

All core scripts in this project are accompanied by dedicated unit tests located in the `tests/` directory. These tests are designed to ensure the correctness, robustness, and stability of the pipeline's components, including:

- Feature extraction and data parsing
- Preprocessing and transformation routines
- Model training and evaluation functions
- Inference logic and performance metrics computation

#### ✅ Running Unit Tests

You can execute the complete test suite from the root directory using the `unittest` framework’s built-in discovery mechanism:

```bash
python -m unittest discover -s tests
```

This command will automatically detect and run all test files matching the `test_*.py` naming convention in the tests/ folder.
---

## ⚙️ Requirements

This section lists all the software and libraries required to run the CMEPDA-EXAM pipeline smoothly.

---

### 🐍 Python Environment

- **Python version:** 3.9 (recommended for compatibility with all packages)

- **Key Python packages:**  
  The pipeline relies on a set of scientific, machine learning, and utility libraries, including:

  - **Numerical and Data Handling:**  
    `numpy`, `pandas`, `scipy`, `scikit-image`

  - **Machine Learning and Deep Learning:**  
    `scikit-learn`, `keras`, `tensorflow`

  - **Neuroimaging Tools:**  
    `nibabel`, `nilearn`

  - **Visualization and Plotting:**  
    `matplotlib`, `seaborn`, `graphviz`

  - **Command-line Argument Parsing:**  
    `argparse`

  - **File System and Utilities:**  
    `pathlib`, `joblib`, `tabulate`, `loguru`

  - **Documentation Tools:**  
    `sphinx`, `sphinx-rtd-theme`, `sphinx-book-theme`

- **MATLAB-Python interface:**  
  `matlab.engine` module is required to run MATLAB scripts from Python for feature extraction.

---

### Installing Python dependencies

To install all necessary Python packages, run the following command from your environment where Python 3.9 is active:

```bash

pip install -r requirements.txt

```

### 🧮 MATLAB Requirements

- **MATLAB version:**  
  The pipeline requires **MATLAB R2024b** or a compatible recent release. This is necessary to run the feature extraction scripts that process the MRI images and extract region-based metrics.

#### MATLAB Engine API for Python

The **MATLAB Engine API for Python** enables calling MATLAB functions directly from Python scripts. This integration allows the pipeline to automate feature extraction without manual intervention, combining the strengths of MATLAB’s neuroimaging tools with Python’s machine learning frameworks.

---

## 🚀 How to Run

The `main.py` script supports **two execution modes**: Training mode and Inference mode.

---

### 1. Training Mode

Runs the full pipeline: extracts features via MATLAB, trains and evaluates the classifier, saves the trained model.

```bash
python main.py \
  --folder_path "/path/to/nifti_folder" \
  --atlas_file "/path/to/original_atlas.nii.gz" \
  --atlas_file_resized "/path/to/resampled_atlas.nii.gz" \
  --atlas_txt "/path/to/atlas_labels.txt" \
  --metadata_csv "/path/to/metadata.csv" \
  --matlab_path "/path/to/MATLAB_folder" \
  --classifier {rf, svm} \
  --n_iter N_ITER \
  --cv CV \
  --kernel {linear, rbf}
```

### 2. Inference Mode
Uses a previously trained model to classify new independent NIfTI images, skipping training.

```bash
python main.py \
  --folder_path "/path/to/nifti_folder" \
  --atlas_file "/path/to/original_atlas.nii.gz" \
  --atlas_file_resized "/path/to/resampled_atlas.nii.gz" \
  --atlas_txt "/path/to/atlas_labels.txt" \
  --metadata_csv "/path/to/metadata.csv" \
  --matlab_path "/path/to/MATLAB_folder" \
  --classifier rf \
  --n_iter 10 \
  --cv 20 \
  --kernel linear \
  --use_trained_model \
  --trained_model_path "/path/to/trained_model.joblib" \
  --nifti_image_path "/path/to/independent_nifti_images"
```

#### Notes

- If the `--use_trained_model` flag is **not** provided, the script runs in **Training Mode**.
- After training, the pipeline is saved as a `.joblib` file.
- In Inference Mode, feature extraction is still performed on the new images using MATLAB, but classification uses the pre-trained model without re-training.
- The pipeline supports three variants of Random Forest:
  - Standard RF
  - RF with PCA
  - RF with RFE (Recursive Feature Elimination)
- ⚠️ **If Random Forest (`--classifier rf`) is selected**, the user is prompted **after feature extraction** but **before training** to choose the desired variant:
  - A terminal input (`yes/no`) will ask whether to apply **PCA**.
  - If PCA is not chosen, a second prompt will ask whether to apply **RFE**.
  - If neither PCA nor RFE is selected, standard Random Forest is used.
- For RFE, the top 8 ROIs are visualized in a pie chart to highlight their relative importance in classification performance.

#### 🧾 Command-Line Parameters Overview

| Parameter              | Description                                     | Required in          |
|------------------------|-------------------------------------------------|----------------------|
| `--folder_path`        | Directory containing subject NIfTI images       | Training only        |
| `--atlas_file`         | Original brain atlas in `.nii` or `.nii.gz`     | Training & Inference |
| `--atlas_file_resized` | Resampled atlas aligned with image dimensions   | Training & Inference |
| `--atlas_txt`          | Text file with ROI labels (one per line)        | Training & Inference |
| `--metadata_csv`       | CSV file with subject IDs and diagnosis labels  | Training & Inference |
| `--matlab_path`        | Folder containing MATLAB scripts                | Training & Inference |
| `--classifier`         | Classifier type: `rf` (Random Forest) or `svm`  | Training only        |
| `--n_iter`             | Number of combinations for parameters search    | Training only        |
| `--cv`                 | Number of K-folds for cross-validation          | Training only        |
| `--kernel`             | SVM kernel type: `linear` or `rbf`              | Training only        |
| `--use_trained_model`  | Enables Inference Mode using a saved model      | Inference only       |
| `--trained_model_path` | Path to a `.joblib` model previously trained    | Inference only       |
| `--nifti_image_path`   | Directory with new subjects to classify         | Inference only       |

---

## 🧠 Pipeline Guide

This project is structured around a modular **4-step pipeline**, combining feature extraction from brain MRI images (via MATLAB) and classification (via Python) for Alzheimer's Disease detection.

---

### 1. 🧪 Feature Extraction (via MATLAB)

The core of the feature extraction process is implemented in MATLAB using the script `feature_extractor.m`. This script operates on brain images in **NIfTI format** (`.nii`, `.nii.gz`) and requires a compatible brain atlas that partitions the brain into **Regions of Interest (ROIs)**.

For each subject and for each ROI defined in the atlas, the script computes the following statistical features:

- **Mean Intensity**: average voxel intensity within the ROI, indicating tissue characteristics.
- **Standard Deviation**: variability of the intensity values within the region.
- **Region Volume**: number of voxels (i.e., size) comprising the ROI.

- ⚠️ Mean and Standard Deviation are set as "default figures of merit"

### 2. 🤖 Classification (via Python)

The Python component handles all tasks related to machine learning. After reading the features extracted in MATLAB, the script can train or apply classification models based on user selection.

Supported classifiers:

#### ✅ Random Forest (`--classifier rf`)
- **Standard Random Forest**: no dimensionality reduction or feature selection.
- **PCA variant**: applies **Principal Component Analysis** before classification to reduce feature dimensionality.
- **RFE variant**: uses **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify the most informative ROIs automatically.

> ℹ️ If Random Forest is selected, the user is interactively prompted in the terminal to choose whether to apply PCA or RFE.

#### ✅ Support Vector Machine (SVM) (`--classifier svm`)
- Uses the `scikit-learn` implementation.
- Supports custom kernel selection (e.g., `--kernel rbf` or `--kernel linear`).
- Works directly with the full feature set.

---

### 📊 3. Performance Metrics

The pipeline provides a comprehensive evaluation of the classification model using multiple performance metrics. These metrics are computed for each cross-validation fold and averaged over `n_iter` independent repetitions to ensure robustness.

- **Accuracy**  
  The proportion of correctly classified samples over the total number of predictions. It gives a general sense of overall model performance.

  (TP + TN) / (TP + TN + FP + FN)

- **Precision**  
  Also known as Positive Predictive Value, it measures the ratio of correctly predicted positive observations to the total predicted positives.

  TP / (TP + FP)
  

- **Recall (Sensitivity / True Positive Rate)**  
  Indicates the model's ability to detect positive instances (e.g., Alzheimer’s patients) correctly.

  TP / (TP + FN)

- **F1 Score**  
  The harmonic mean of Precision and Recall. It balances both metrics and is especially useful when classes are imbalanced.

  2 x Precision x Recall / (Precision + Recall)

- **Specificity (True Negative Rate)**  
  Measures the proportion of correctly identified negative cases (e.g., healthy controls).

  TN / (TN + FP)
  

- **AUC (Area Under the ROC Curve)**  
  Represents the model's ability to distinguish between classes across all possible thresholds. A higher AUC indicates a better-performing classifier.

- **Confidence Intervals**  
  Each metric is reported along with a 95% confidence interval:
  
  - **Binomial-based intervals** for accuracy, precision, recall, and specificity.
  - **Bootstrap-based intervals** for more complex metrics such as AUC and F1 score.

These intervals are computed over multiple repetitions of cross-validation and help quantify the uncertainty of the model’s performance.

---

### 📦 4. Output Artifacts

After the script completes execution, the following outputs are generated:

- **📋 Tabulated Metrics Summary**  
  A table summarizing all key metrics for each iteration is printed in the terminal. This allows transparent comparison across runs.

- **📈 Visualization Outputs**

  1. **ROC Curve**  
     Displays the trade-off between true positive rate and false positive rate for all classification thresholds. Useful for visual inspection of model discrimination power.

  2. **Performance Bar Chart**  
     A bar plot comparing mean values (with error bars for confidence intervals) of each metric such as Accuracy, Precsion, Recall, F1 Score, Sensitivity and AUC.

  3. **Feature Importance Plot**  
     (Only available if using Random Forest with RFECV)  
     Visualizes the most relevant features selected by the Recursive Feature Elimination process, ranked by importance.

- **💾 Trained Model Persistence**  
  The final classifier, including any dimensionality reduction steps (e.g., PCA or RFECV), is serialized and saved in a `.joblib` file. This file can later be reused for inference without repeating the entire training pipeline.

---



## Documentation
INSERIRE LINK DOCUMENTAZIONE

--- 

## 📄 License
This project is released **exclusively for academic, educational, and non-commercial research purposes**. All code, data processing routines, and related materials are provided as is, without any warranty of fitness for a particular purpose or responsibility for potential misuse.

The intellectual property of this work remains with the authors and contributors. Users are granted permission to view, modify, and use the codebase solely for:

- Research within academic or scientific institutions

- Course assignments, theses, and publications (with proper attribution)

- Reproducibility and methodological benchmarking

⚠️ Restrictions
- **Commercial use is strictly prohibited** without prior written consent from the authors.

- Redistribution of this work or derivative versions must retain this license and clearly indicate any modifications made.

- Any results or publications derived from the use of this project must cite the original authors appropriately.

By using this repository, you agree to comply with the terms outlined above. For clarification or collaboration inquiries, please contact the repository maintainers and authors.
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

---

## ⭐ Contributing

We welcome contributions that help improve or extend the functionality of this project. Whether you're fixing a bug, adding new features, improving documentation, or optimizing existing code, your input is highly appreciated.

### 🔧 How to Contribute

Follow these steps to propose your contribution:

1. **Fork** the repository to your own GitHub account.  
2. **Create a new branch** for your feature or fix:  
   ```bash
   git checkout -b your-feature-name
   ```
3. **Make your changes locally**, ensuring code quality and proper documentation.
4. **Commit your changes with a clear and concise message:**
   ```bash
   git commit -am "Add feature: [short description]"
  ```
5. **Push to your fork:**
    ```bash
      git push origin your-feature-name
    ```
6. **Open a Pull Request** from your branch to the main branch of the original repository. Please include:

  - A detailed explanation of your changes

  - References to related issues (if applicable)

  - Any additional context or testing steps



  
