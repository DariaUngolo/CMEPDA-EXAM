# 🧠 Scalable MRI-Based Classification of Alzheimer’s Disease with Classical and Deep Learning

**Brain MRI Classification Pipeline for Alzheimer’s Disease Detection**

This project focuses on the development and implementation of a **binary classifier** aimed at distinguishing between subjects diagnosed with **Alzheimer’s Disease (AD)** and **healthy control subjects (CTRL)**. The dataset used to train the ready-to-use models included **333 brain MRI scans**, comprising **144 subjects diagnosed with Alzheimer's Disease (AD)** and **189 healthy control subjects (CTRL)**.
However, users are free to use **any other dataset** for training or inference, as long as it conforms to the **expected input format and structure** (e.g., NIfTI images and compatible metadata for classification).

The available data includes **3D brain MRI images in NIfTI format**, alongside **two different brain atlases** used to parcellate the brain into anatomically meaningful regions known as **Regions of Interest (ROIs)**. These atlases segment the brain into **56** and **246 ROIs**, respectively, providing different levels of spatial resolution.  
Each brain atlas is also accompanied by a **look-up table (LUT)** that lists the names of all ROIs along with their corresponding integer labels. This table is essential for identifying and interpreting each brain region during feature extraction and analysis.

## 📂 Repository Tree

```text
.
├── data/
│   ├── test_subject_1/
│   │   └── subject1.nii.gz
│   ├── test_subject_2/
│   │   └── subject2.nii.gz
│   ├── lpba40_atlas.nii.gz
│   ├── BN_atlas.nii.gz
│   ├── lpba40_LUT.txt
│   ├── BN_LUT.txt
│   └── metadata.csv
│
├── docs/
│   └── ...documentation files...
│
├── tests/
│   └── ...unit tests...
│
├── ML_codes/
│   └── ...machine learning modules...
│
├── CNN_codes/
│   └── ...deep learning modules...
│
├── main/
│   ├── ML_main.py  
│   └── CNN_main.py
│
├── trained_models/
│   ├── CNN_trained_model.h5
│   └── trained_model*.joblib  #one for each type of classifiers
│
├── plots and images/
│
└── README.md
```

### Images and Atlas Description 

The MRI scans used in this project are stored in the **NIfTI format** (`.nii` or `.nii.gz`), which is a widely adopted standard for medical imaging. This format preserves 3D anatomical structure and supports spatial metadata (e.g., voxel dimensions, orientation, and affine transforms), making it ideal for neuroimaging analysis.

The images have been **preprocessed and segmented** using the **SMWC1** method, which stands for:

> **Segmented, Modulated, and warped Gray Matter Class 1**

This segmentation process is commonly performed via **SPM (Statistical Parametric Mapping)** and involves the following steps:
- **Segmentation**: Identifies gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF) tissue classes.
- **Modulation**: Preserves original volume information by adjusting voxel intensities according to local deformation during spatial normalization.
- **Warping**: Aligns individual subjects' anatomy into a common reference space (e.g., MNI152), enabling group-level comparisons.

The resulting `smwc1*.nii` image contains **gray matter tissue probability maps**, where each voxel value represents the **likelihood (between 0 and 1)** of being gray matter, adjusted for individual brain volume and spatial normalization.

##### 🔍 Visualization

These NIfTI images can be viewed using any neuroimaging viewer that supports 3D volumes, such as:

- [**Mango**](http://ric.uthscsa.edu/mango/)
- **MRIcron**
- **FSLeyes**
- **ITK-SNAP**

These tools allow you to inspect the anatomical structure, overlay atlas labels, and verify alignment and ROI masking.

Below is a sample slice from an `smwc1` image showing gray matter segmentation in MNI space, overlayed with atlas-based ROIs:

![Example gray matter segmentation](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/brain_example.png)

#### 🧠 Available Atlases

The pipeline supports **multiple brain atlases** to perform ROI-based feature extraction. Each atlas defines a **parcellation scheme** over the brain, assigning each voxel to a specific anatomical or functional region. These parcellations are critical for aggregating voxel-level data (e.g., intensity values) into meaningful region-level statistics used in machine learning.

Two atlases are currently supported and included in the `data/` folder:

1. **LONI Probabilistic Brain Atlas (LPBA40)**
   - **Regions**: 56 anatomical regions  
   - **Format**: Probabilistic, converted to a maximum-probability label map  
   - **Origin**: Developed at the Laboratory of Neuro Imaging (LONI)  
   - **Purpose**: Suitable for coarse anatomical feature aggregation  
   - **Space**: Already coregistered to the MNI152 template space

![Superposition of LONI probabilistic atlas on grey matter](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/brain_atlas_56_spectrum.png)


2. **Brainnetome Atlas**
   - **Regions**: 246 fine-grained regions (210 cortical + 36 subcortical)  
   - **Origin**: Developed by the Chinese Academy of Sciences  
   - **Purpose**: Provides high-resolution parcellation ideal for detecting subtle changes in specific brain circuits  
   - **Space**: Aligned with the MNI152 coordinate system

![Superposition of Brainnetome Atlas on grey matter](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/brain_atlas_246_spectrum.png)

Both atlases are distributed in **NIfTI (.nii.gz)** format and are compatible with the T1-weighted input scans. Since the atlases are pre-aligned to the **MNI152 standard space**, they ensure anatomical consistency with most neuroimaging datasets without requiring additional registration steps.

**Automatic Resampling**

If the atlas resolution does not match the input image (e.g., different voxel sizes or matrix dimensions), the pipeline automatically **resamples the atlas** to match the subject's MRI geometry using nearest-neighbor interpolation. This guarantees voxelwise correspondence and preserves the integrity of region labels.

The resampled atlas is saved in the output directory and reused in subsequent runs to avoid redundant computation.

### Feature Extraction (ML)

Feature extraction is performed using **MATLAB**, an essential step in the machine learning approach, which processes the MRI scans and atlas-based segmentations to compute region-level statistics for each ROI. Specifically, the extraction pipeline calculates:

- **Mean intensity**
- **Standard deviation**
- **Region volume** (i.e., number of voxels)

These features serve as the input to the classification pipeline implemented in **Python**.

### Classification Approaches

The binary classification task is tackled using two complementary approaches:

1. **Classical Machine Learning**, including:
   - **Random Forest (RF)** classifier, with three variants:
     - Standard Random Forest
     - Random Forest combined with **Principal Component Analysis (PCA)** for dimensionality reduction
     - Random Forest with **Recursive Feature Elimination (RFE)** for automated feature selection
   - **Support Vector Machine (SVM)** with customizable kernels (e.g., linear, RBF)

2. **Deep Learning**, implemented through a **3D Convolutional Neural Network (CNN)**, which learns hierarchical features directly from the MRI volumes.

### Advanced ROI-based Analysis (ML)

The ROIs identified as most informative by RFE are used in the second phase of the project to refine image processing. Specifically, these top-ranked ROIs define a **bounding box** around the brain, which is then used to **crop the MRI volumes** to focus on the most diagnostically relevant areas. This localized cropping facilitates further analysis and potentially improves the deep learning model’s ability to focus on pathological patterns linked to Alzheimer’s Disease.

### Evaluation Methodology 

For each classical ML classifier configuration, the model is trained and evaluated over **10 independent runs**, using a robust **15-fold cross-validation** strategy to ensure statistical reliability and generalizability. Performance metrics are **averaged across runs and folds**, and include:

- *Accuracy*
- *Precision*
- *Recall*
- *F1 score*
- *Sensitivity*
- *Area Under the ROC Curve (AUC)*

Results are visualized through:

- **ROC curves** summarizing classifier discrimination ability
- **Bar charts** illustrating mean performance metrics along with their confidence intervals

In the case of the **RFE-based Random Forest**, an additional **pie chart** is generated to display the **top 8 most relevant ROIs** contributing to the classification decision, along with their relative feature importances.


For the **deep learning (CNN) approach**, the model is trained for **150 epochs** with a **batch size of 32** and an initial **learning rate of 0.001**. Two callbacks are used during training:

- **EarlyStopping** to halt training when performance stops improving  
- **ReduceLROnPlateau** to lower the learning rate when validation performance plateaus

Evaluation of the CNN is based on:

- Accuracy  
- Recall  
- AUC

Throughout training, the model logs per-epoch values of these metrics on the **training**, **validation**, and **test** sets. After training, the following outputs are generated automatically:

- Accuracy and loss curves for training and validation  
- ROC curves for validation and test datasets

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

### 🧾 Logging and Debugging

The pipeline uses the `loguru` package for structured and user-friendly logging. Logs include:

- Information on each pipeline step (feature extraction, training, evaluation)
- Warnings for misaligned input images or missing files
- Errors with detailed traceback for debugging

---

## ⚙️ Requirements

This section lists all the software and libraries required to run the CMEPDA-EXAM pipeline smoothly.

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

## 🧠 ML Pipeline Guide

This project is structured around a modular **4-step pipeline**, combining feature extraction from brain MRI images (via MATLAB) and classification (via Python) for Alzheimer's Disease detection.

### 1. 🧪 Feature Extraction (via MATLAB)

The core of the feature extraction process is implemented in MATLAB using the script `feature_extractor.m`. This script operates on brain images in **NIfTI format** (`.nii`, `.nii.gz`) and requires a compatible brain atlas that partitions the brain into **Regions of Interest (ROIs)**.

For each subject and for each ROI defined in the atlas, the script computes the following statistical features:

- **Mean Intensity**: average voxel intensity within the ROI, indicating tissue characteristics.
- **Standard Deviation**: variability of the intensity values within the region.
- **Region Volume**: number of voxels (i.e., size) comprising the ROI.

- ⚠️ Mean and Standard Deviation are set as "default figures of merit" but one can switch to every other possible combination according to their preferences

The atlas is overlaid on the input MRI as a **binary mask**, and statistics are computed **only within voxels labeled as part of each ROI**. To reduce the impact of background noise or interpolation artifacts, a small intensity **threshold of 10⁻⁶** is applied: any voxel with an intensity value below this threshold is ignored during feature computation.

This masking and thresholding step ensures that the extracted features are robust, biologically meaningful, and not corrupted by out-of-brain or near-zero intensity values.

The voxels above the chosen threshold do not cause the loss of brain regions, as can be seen in the image below, where voxels exceeding the threshold are highlighted in white.


![voxel above threshold](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/smwc1AD_1_colored.png)

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

  ![metrics scheme](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/metrics_scheme.png)

- **AUC (Area Under the ROC Curve)**
The AUC measures the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one. It summarizes the model’s ability to distinguish between classes across all classification thresholds. An AUC of 0.5 indicates no discriminative power (equivalent to random guessing), while an AUC of 1.0 indicates perfect discrimination. A higher AUC thus reflects a more effective and statistically robust classifier, especially in imbalanced classification tasks.

![ROC-AUC](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/AUC-ROC%20example.png)

- **Confidence Intervals**
Each performance metric is reported along with a 95% confidence interval to provide a measure of statistical reliability and variability:

  - **Binomial-based intervals** are used for metrics derived from count-based proportions (e.g., accuracy, precision, recall, specificity). These intervals are computed assuming a binomial distribution, and reflect the uncertainty due to the finite sample size.

  - **Bootstrap-based intervals** are applied to metrics that are not simple proportions—such as AUC and F1 score—by repeatedly resampling the data with replacement and recalculating the metric. The resulting distribution allows for a non-parametric estimation of the confidence interval, making it more flexible and robust when analytical solutions are not available.

These intervals help assess the stability and generalizability of the model's performance, rather than relying solely on point estimates.


### 📦 4. Outputs

After the script completes execution, the following outputs are generated:

- **📋 Tabulated Metrics Summary**  
  A table summarizing all key metrics for each iteration is printed in the terminal. This allows transparent comparison across runs.
  The results are displayed as in the table below:

| **Metric**    | **Score** | **± Error** |
|---------------|-----------|-------------|
| Accuracy      | 0.82      | ±0.07       |
| Precision     | 0.67      | ±0.08       |
| Recall        | 0.91      | ±0.05       |
| F1-score      | 0.77      | ±0.07       |
| Specificity   | 0.78      | ±0.07       |
| AUC           | 0.84      | ±0.08       |

- **📈 Visualization Outputs**

  1. **ROC Curve**  
     Displays the trade-off between true positive rate and false positive rate for all classification thresholds. Useful for visual inspection of model discrimination power.

![ROC curve and AUC example](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/ROC_rf_RFE_100_15_mean%2Bstd_BN.png)

  2. **Performance Bar Chart**  
     A bar plot comparing mean values (with error bars for confidence intervals) of each metric such as Accuracy, Precsion, Recall, F1 Score, Sensitivity and AUC.
     
![Bar chart example](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/metrics_rf_RFE_100_15_mean%2Bstd_BN.png)


  3. **Feature Importance Plot**  
     (Only available if using Random Forest with RFECV)  
     Visualizes the most relevant features selected by the Recursive Feature Elimination process, ranked by importance.
     
![Pie chart example](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/piechart_rf_RFE_100_15_mean%2Bstd_BN.png)

- **💾 Trained Model Persistence**
The final classifier — including any dimensionality reduction steps (e.g., PCA or RFECV) — is serialized and saved in a .joblib file. Among all models trained during cross-validation, the one corresponding to the **median AUC** is selected and saved to ensure robust statistical performance. This file can later be reused for inference without repeating the entire training pipeline.

- **🧠 Prediction Output**

When applying a trained model to an independent test image (e.g., from an external dataset), the pipeline returns a **prediction table** with the following information for each subject:

- **`Label`**: The predicted class, where:
  - `0` indicates a healthy subject (control),
  - `1` indicates a subject classified as having Alzheimer’s disease.

- **`Probability`**: The confidence score (a float between 0 and 1) associated with the prediction. This represents the model's estimated probability that the predicted label is correct.

The output allows for both binary classification and an assessment of prediction confidence, which can be used for thresholding or uncertainty-based analysis.


| Subject ID | Label | Probability |
|------------|-------|-------------|
| sub-001    | 1     | 0.87        |


- In this example:
  - `sub-001` is predicted as having Alzheimer’s disease with high confidence.


---
## 🧠 CNN Pipeline Guide

This project is structured around a modular **6-step pipeline** that includes preprocessing, augmentation, model training, and interactive classification of brain MRI data.

### 1. 🔍 GPU Configuration

Before any processing begins, the script detects available GPUs and configures memory growth to optimize usage. If no GPU is detected, it defaults to CPU execution.

- **Purpose**: Ensures compatibility with various hardware configurations and leverages GPU acceleration when available.
- **Implemented by**:
  - `tf.config.list_physical_devices('GPU')`
  - GPU memory growth setup.


### 2. 🧪 Preprocessing

In the deep learning approach, it is **not necessary to use ROIs as binary masks** for preprocessing. Instead, each MRI scan is **cropped along the z-axis around the most relevant region of interest**, specifically the hippocampus. This region has been identified as the most important for classification in our case, which aligns with known medical findings regarding neurodegenerative diseases.

Since convolutional neural networks (CNNs) require input images to have consistent dimensions, each cropped volume is **padded** to match the size of the largest cropped sample in the dataset.

The processed images are saved as **3D NumPy arrays** with a single intensity channel, making them directly compatible with CNN architectures.

Voxel intensities are **not normalized between 0 and 1 by default**, but a built-in normalization function is available for users who wish to apply it.

**Output**: A 4D numpy array of preprocessed images and corresponding labels.

### 3. 🎨 Data Augmentation

Given the limited size of our dataset, we implement a **data augmentation strategy** to improve the model's generalization and reduce overfitting.

The following augmentations are applied randomly to the training set:
- **Random intensity variation**
- **Random crop-and-zoom**

As a result, the training dataset is **tripled**, consisting of:
1. Original cropped and padded images
2. Images with random crop-and-zoom transformations
3. Images with random intensity modifications

This process enriches the diversity of the training data and strengthens the performance of the CNN on unseen samples.

**Output**: Augmented training data with enhanced diversity.


### 4. 📂 Data Splitting

Splits the dataset into three subsets:

- **Training Set**: Used to train the CNN.
- **Validation Set**: Monitors performance during training.
- **Test Set**: Independently evaluates the model after training.

**Output**: Training, validation, and test datasets.


### 5. 🤖 CNN Training

Trains a Convolutional Neural Network (CNN) model:

- **Model Architecture**: Created using the `MyCNNModel` class.  
    This is a 3D CNN designed for volumetric data. It consists of four convolutional blocks followed by a classification head.

    **Architecture:**
    - **Block 1**  
      ` Conv3D(8) → ReLU → BatchNorm →MaxPooling3D → Dropout(0.1)`

    - **Block 2**  
      `Conv3D(16) → ReLU → BatchNorm →MaxPooling3D →  Dropout(0.2)`

    - **Block 3**  
      `Conv3D(32) → ReLU  → MaxPooling3D → Dropout(0.2)`

    - **Block 4**  
      `Conv3D(32) → ReLU → Dropout(0.2)`

    - **Classification Head**  
      `Flatten → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)`

      The model uses L2/L1 regularization, ReLU activations, and pooling to reduce dimensionality and prevent overfitting.

    - **Input Shape**: Derived automatically from the preprocessed data.

- **Training Configuration**: Parameters such as epochs and batch size are customizable via command-line arguments.

- **Output**: A trained model saved as a `.h5` file.


### 6. 🧪 Interactive Classification

After training, the user can optionally classify new NIfTI images using the trained model. Key steps:

1. **Model Loading**: Loads the pre-trained model for inference.
2. **Preprocessing**: Ensures the new image matches the input shape expected by the model.
3. **Prediction**: Outputs the class probabilities and the predicted label.
4. **Interactive Loop**: Allows users to classify additional images in a session.

**Output**: A tabulated summary of predictions with probabilities for each classified image.



### 🛠 Outputs

After executing the pipeline, the following are produced:

1. **Trained Model**: Saved as `trained_model.h5` for future use.
2. **Classification Results**: A tabulated summary of predictions during interactive classification.
3. **Performance Logs**: Detailed logs, including GPU usage and intermediate steps, to assist debugging and performance analysis.


- **📈 Visualization Outputs**

1. **📋 Tabulated Metrics Summary**  
  A table summarizing all key metrics for validation data and test data:

| **Val_Metric**    | **Score** | **± Error** |
|---------------|-----------|-------------|
| Val_Accuracy      | 0.68     | ± 0.07         |
| Val_Recall        |  0.60      | ± 0.07      |
| Val_AUC           |  0.76      | ± 0.07       |
| Val_ROC           |  0.76      | ±       |

| **Test_Metric**    | **Score** | **± Error** |
|---------------|-----------|-------------|
| Test_Accuracy      | 0.72      | ± 0.06         |
| Test_Recall        |  0.79        | ± 0.06        |
| Test_AUC           |  0.8        | ± 0.06       |
| Test_ROC           |  0.8        | ±       |

2. **Training and Validation Performance (Plot)**  
   This figure shows both **Loss** and **AUC** curves during training and validation.  
   The top subplot compares the training and validation AUC across epochs, while the bottom subplot compares the corresponding Loss. This visualization helps identify potential overfitting or underfitting during model training.

![AUC and Loss during training and validation](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/AUC_loss_train%2Bval.png) 
3. **Validation ROC Curve**  
   Displays the Receiver Operating Characteristic (ROC) curve on the validation set.  
   This curve helps evaluate how well the model distinguishes between classes before final testing.

![Validation ROC Curve](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/CNN_validation_roc.png)

4. **Test ROC Curve**  
   Shows the ROC curve obtained on the test set.  
   It's a final indicator of model generalization, reflecting performance on unseen data.

![Test ROC Curve](https://github.com/DariaUngolo/CMEPDA-EXAM/blob/main/plots%20and%20images/CNN_test_roc.png)

- **🧠 Prediction Output**

When applying a trained model to an independent test image (e.g., from an external dataset), the pipeline returns a **prediction table** with the following information for each subject:

- **`Label`**: The predicted class, where:
  - `0` indicates a healthy subject (control),
  - `1` indicates a subject classified as having Alzheimer’s disease.
  - 
| Subject ID | Label | Probability |
|------------|-------|-------------|
| sub-001    | 1     | 0.99        |


- In this example:
  - `sub-001` is predicted as having Alzheimer’s disease with high confidence.
---
## 🚀 How to Run

> 🧭 **Important:** You must run the script from the **root directory of the project** using a terminal.


**The `ML_main.py` script supports two execution modes**: Training mode and Inference mode.


### 1. ML Training Mode

Runs the full pipeline: extracts features via MATLAB, trains and evaluates the classifier, saves the trained model.

```bash
python ML_main.py \
  --folder_path "/path/to/nifti_folder" \
  --atlas_file "/path/to/original_atlas.nii.gz" \
  --atlas_file_resized "/path/to/resampled_atlas.nii.gz" \
  --atlas_txt "/path/to/atlas_labels.txt" \
  --metadata_csv "/path/to/metadata.csv" \
  --matlab_path "/path/to/MATLAB_folder" \
  --classifier {rf, svm} \
  --n_iter <number_of_combinations> \
  --cv <number_of_folds> \
  --kernel {linear, rbf}
```

### 2. ML Inference Mode
Uses a previously trained model to classify new independent NIfTI images, skipping training.

```bash
python ML_main.py \
  --atlas_file_resized "/path/to/resampled_atlas.nii.gz" \
  --atlas_txt "/path/to/atlas_labels.txt" \
  --matlab_path "/path/to/MATLAB_folder" \
  --use_trained_model \
  --trained_model_path "/path/to/trained_model.joblib" \
  --nifti_image_path "/path/to/independent_nifti_images"
```
> ⚠️ **Warning:** The following models have been trained using the **mean_std** feature set derived from the  **LPBA40**.  
> To ensure compatibility and accurate predictions, you **must use the LPBA40** during feature extraction.  
> Using a different atlas may lead to incorrect results.

| Model Name                        | Description                     |
|----------------------------------|---------------------------------|
| `trained_model_rf.joblib`        | Random Forest                   |
| `trained_model_rf_pca.joblib`    | Random Forest + PCA             |
| `trained_model_rf_rfecv.joblib`  | Random Forest + RFECV           |
| `trained_model_svm_linear.joblib`| SVM with Linear Kernel          |
| `trained_model_svm_rbf.joblib`   | SVM with RBF Kernel             |



#### ML Notes

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

#### 🧾 ML Command-Line Parameters Overview

| Parameter              | Description                                     | Required in          |
|------------------------|-------------------------------------------------|----------------------|
| `--folder_path`        | Directory containing subject NIfTI images       | Training only        |
| `--atlas_file`         | Original brain atlas in `.nii` or `.nii.gz`     | Training & Inference |
| `--atlas_file_resized` | Resampled atlas aligned with image dimensions   | Training & Inference |
| `--atlas_txt`          | Text file with ROI labels (one per line)        | Training & Inference |
| `--metadata_csv`       | CSV file with subject IDs and diagnosis labels  | Training only        |
| `--matlab_path`        | Folder containing MATLAB scripts                | Training & Inference |
| `--classifier`         | Classifier type: `rf` (Random Forest) or `svm`  | Training only        |
| `--n_iter`             | Number of combinations for parameters search    | Training only        |
| `--cv`                 | Number of K-folds for cross-validation          | Training only        |
| `--kernel`             | SVM kernel type: `linear` or `rbf`              | Training only        |
| `--use_trained_model`  | Enables Inference Mode using a saved model      | Inference only       |
| `--trained_model_path` | Path to a `.joblib` model previously trained    | Inference only       |
| `--nifti_image_path`   | Directory with new subjects to classify         | Inference only       |

---

**The `CNN_main.py` script supports two execution modes**: Training mode and Inference mode.


## 1. CNN Training Mode

This mode trains a CNN model using the provided dataset and saves the resulting trained model for future inference.

To handle the computational demands of 3D convolutional neural networks for Alzheimer’s classification,  
we leveraged **GPU** acceleration. GPUs excel at parallel processing,  
which significantly speeds up both data augmentation and model training.  

By using **TensorFlow**’s GPU capabilities, we optimized the processing of large volumetric medical images,  
substantially reducing training times.  

**It is recommended to run this code on a machine equipped with a GPU  
to fully benefit from these optimizations and accelerate the training process.**

```bash
python CNN_main.py \
  --image_folder "/path/to/nifti_folder" \
  --atlas_path  "/path/to/original_atlas.nii.gz" \
  --metadata "/path/to/metadata.csv" \
  --epochs <number_of_epochs> \
  --batchsize <batch_size>
```

## 2. CNN inference Mode

Uses a pre-trained CNN model to classify new independent NIfTI images.

```bash
python CNN_main.py \
  --atlas_path  "/path/to/original_atlas.nii.gz" \
  --nifti_image_path "/path/to/nifti_image" \
  --use_trained_model \
  --trained_model_path "/path/to/trained_model.h5"
```

#### CNN Notes

- Ensure that the atlas file aligns with the resolution of the NIfTI images used for both training and inference.
- In Training Mode, the `--metadata` file must include all necessary labels for proper model training.
- The script supports dynamic batch sizes and epochs; experiment with these parameters to optimize training performance.
- Pre-trained models must match the input data format and preprocessing pipeline to avoid compatibility issues during inference.
- Inference Mode allows classification of a single image at a time; batch processing requires script modification.
- If using a custom atlas, ensure it is preprocessed and compatible with the input data structure.

#### 🧾 CNN Command-Line Parameters Overview

| Parameter              | Description                                    | Required in    |
| ---------------------- | ---------------------------------------------- | -------------- |
| `--image_folder`       | Directory containing NIfTI images              | Training only  |
| `--atlas_path`         | Path to the NIfTI atlas file                   | Both Modes     |
| `--metadata`           | Path to a CSV file with metadata and labels    | Training only  |
| `--epochs`             | Number of training epochs                      | Training only  |
| `--batchsize`          | Batch size for model training                  | Training only  |
| `--use_trained_model`  | Enables Inference Mode using a saved model     | Inference only |
| `--trained_model_path` | Path to a `.h5` file for the trained model     | Inference only |
| `--nifti_image_path`   | Path to a NIfTI image for classification       | Inference only |
---
## ⚖️ Conclusions: Comparing Machine Learning and Deep Learning Approaches

This project implements and compares two distinct strategies for classifying Alzheimer's Disease from structural brain MRI: classical **Machine Learning (ML)** based on handcrafted features and **Deep Learning (DL)** based on raw 3D images. Each approach offers specific advantages depending on the dataset size, the interpretability needs, and the computational resources available.

### ✅ Machine Learning

- **Pros**:
  - Highly interpretable: feature importance and selected ROIs offer direct neuroscientific insight.
  - Efficient with small to medium datasets.
  - Easily adaptable to different brain atlases and custom ROI-level features.
- **Cons**:
  - Relies on accurate feature extraction and brain parcellation.
  - Performance plateaus with increasing data complexity.

### 🧠 Deep Learning (CNN 3D)

- **Pros**:
  - Learns hierarchical representations directly from raw volumetric data.
  - High accuracy when trained on sufficiently large and well-preprocessed datasets.
  - Requires minimal manual feature engineering.
- **Cons**:
  - Less interpretable: learned filters are hard to map to anatomical meaning.
  - Demands larger training sets and higher computational power (especially GPUs).
  - Sensitive to overfitting on small datasets.

### 📊 Summary

| Method             | Input Type         | Preprocessing | Interpretability | Performance | Data Needs  |
|--------------------|--------------------|---------------|------------------|-------------|-------------|
| Random Forest / SVM | ROI features (CSV) | Atlas-based   | High             | Good        | Low–Medium  |
| 3D CNN             | SMWC1 MRI volumes  | NIfTI-based   | Low              | High (if trained well) | High |

Ultimately, the choice between ML and DL depends on the **goal of the analysis**:

- Use **ML** when explainability and anatomical interpretability are critical (e.g., biomarker identification).
- Use **DL** when performance is paramount and sufficient data and GPU resources are available.

> 🧪 For optimal results, a hybrid approach combining both techniques — e.g., feature-based ML models enhanced by CNN-learned features — may offer the best trade-off between accuracy and interpretability.

---
## 📄 Documentation
[Link Documentazione](https://cmepda-exam-fisica.readthedocs.io/en/latest/index.html)

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
## 📚 References

This project integrates insights, tools, and techniques from both the neuroimaging and machine learning communities. Below is a curated list of key references that guided the design, development, and evaluation of the pipeline.

### 🧠 Neuroimaging and Brain Atlases

1. Retico, Alessandra, et al. (2015). *Predictive Models Based on Support Vector Machines:
Whole-Brain versus Regional Analysis of Structural MRI
in the Alzheimer’s Disease*. **J Neuroimaging**, 25:552-563.  
   [DOI: 10.1111/jon.12163]

2. Sarraf, S, et al. (2017). *DeepAD: Alzheimer’s Disease Classification via Deep Convolutional Neural Networks using MRI and fMRI*. **BioRxiv preprint**, 1-32. [DOI: https://doi.org/10.1101/070441]

3. The **atlases** are: *BN_Atlas_246_2mm.nii.gz* from [https://atlas.brainnetome.org/] & *lpba40_56.nii.gz* from [https://www.loni.usc.edu/research/atlases]


### 🤖 Machine Learning and Deep Learning

4. Breiman, L. (2001). *Random Forests*. **Machine Learning**, 45(1), 5–32.  
   [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

5. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. **Machine Learning**, 20, 273–297.  
   [https://doi.org/10.1007/BF00994018](https://doi.org/10.1007/BF00994018)

6. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep learning*. **Nature**, 521(7553), 436–444.  
   [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539)


### 🧰 Tools and Frameworks

7. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. **Computing in Science & Engineering**, 9(3), 90–95.  
   [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

8. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. **Journal of Machine Learning Research**, 12, 2825–2830.  
   [http://jmlr.org/papers/v12/pedregosa11a.html](http://jmlr.org/papers/v12/pedregosa11a.html)


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



  
