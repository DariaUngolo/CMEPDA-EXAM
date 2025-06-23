import sys
from pathlib import Path
import os

# Add the parent directory of the current working directory to the system path.
sys.path.insert(0, str(Path(os.getcwd()).parent))
import random 


# Import essential libraries for Machine Learning
import numpy as np
import graphviz
import matplotlib.pyplot as plt

# Add Graphviz path to system PATH for Python
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from scipy import stats
from scipy.stats import randint
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFECV

import logging



# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from ML_codes.performance_scores import compute_binomial_error, evaluate_model_performance, compute_roc_and_auc, plot_roc_curve, compute_average_auc, compute_mean_std_metric, plot_performance_bar_chart # Importing a custom module for performance evaluation

# Importing a custom module to interact with MATLAB EnginE
from ML_codes.feature_extractor import feature_extractor


# Hyperparameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 500),  # number of trees
    'max_depth': randint(5, 50),        # maximum depth of the tree
    'min_samples_split': randint(5, 15),  # minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 5),   # minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'],    # number of features to consider when looking for the best split
    'bootstrap': [True, False]           # whether bootstrap samples are used when building trees
}



class MetricsLogger:
    """
    A class to log and compute metrics for model evaluation.

    Attributes
    ----------
    mean_fpr : np.ndarray
        Array of evenly spaced false positive rates for ROC computation.

    accuracy_list, precision_list, recall_list, f1_list, specificity_list : list
        Lists to store metric values for each iteration.

    auc_list : list
        List to store AUC values for each iteration.

    tpr_list : list
        List to store interpolated true positive rates for mean ROC computation.

    model_auc_pairs : list
        List of tuples containing models and their corresponding AUC values.

    selected_rois_list : list
        (Optional) List of selected ROI feature names corresponding to each model.
        If not provided, entry is set to None.
    """

    def __init__(self, mean_fpr):
        self.mean_fpr = mean_fpr
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.specificity_list = []
        self.auc_list = []
        self.tpr_list = []
        self.model_auc_pairs = []
        self.selected_rois_list = []  # Optional ROI names per model

    def append_metrics(self, metrics_scores, y_tst, y_prob, model, selected_rois=None):
        """
        Append metrics, compute ROC/AUC, and store model information.

        Parameters
        ----------
        metrics_scores : dict
            Dictionary containing evaluation metrics (e.g., accuracy, precision).

        y_tst : np.ndarray
            Ground truth labels for the test set.

        y_prob : np.ndarray
            Predicted probabilities for the test set.

        model : sklearn estimator
            The trained model to be logged.

        selected_rois : list or np.ndarray, optional
            List of ROI feature names used for training the model.
            If not applicable, pass None (default).
        """
        self.accuracy_list.append(metrics_scores['Accuracy'])
        self.precision_list.append(metrics_scores['Precision'])
        self.recall_list.append(metrics_scores['Recall'])
        self.f1_list.append(metrics_scores['F1-score'])
        self.specificity_list.append(metrics_scores['Specificity'])

        # Compute ROC and AUC
        fpr, tpr, roc_auc, auc_err = compute_roc_and_auc(y_tst, y_prob)
        self.auc_list.append(roc_auc)

        # Interpolate TPR for mean ROC computation
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tpr_list.append(interp_tpr)

        # Save model and AUC
        self.model_auc_pairs.append((model, roc_auc))

        # Save selected ROI names if provided
        self.selected_rois_list.append(selected_rois)


       


    

def RFPipeline_noPCA(df1, df2, n_iter, cv):
    """
    
    Train and evaluate a Random Forest classifier pipeline without PCA.

    The function performs 10 iterations of training and evaluation:

        - Converts DataFrames to numpy arrays and encodes labels as binary (0, 1).
        - Defines a hyperparameter search space for Random Forest.
        - Splits data into train/test sets (90%/10%).
        - Optimizes hyperparameters using RandomizedSearchCV.
        - Computes performance metrics and ROC curves.
        - Aggregates results and plots mean ROC and metrics.
        - Returns the model from the iteration with median AUC.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Feature matrix with input variables.
    df2 : pandas.DataFrame
        Target labels with categorical classes (e.g., 'Normal', 'AD').
    n_iter : int
        Number of parameter settings sampled by RandomizedSearchCV.
    cv : int
        Number of folds in cross-validation during hyperparameter tuning.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained Random Forest pipeline from the iteration with median AUC.

    Notes
    -----
    - No PCA is applied; original features are used.
    - Labels mapped as 'Normal' → 0, 'AD' → 1.
    - Random Forest uses balanced class weights.
    - Two iterations reduce variance in performance estimation.
    - Parallel processing uses all CPU cores during hyperparameter search.

    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    
    """

    
    # Convert features and labels to numpy arrays
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values # convert labels to binary


    # Get column names to be used as feature names for visualization
    region = list(df1.columns.values)
    

    mean_fpr = np.linspace(0, 1, 100) 
    metrics_logger = MetricsLogger(mean_fpr)

    for iteration in range(10):
        
        logging.info(f"Iteration {iteration + 1} - Splitting data into train and test sets.")
        
        # Split data into training and test sets (10% test data)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1)

        # Define a pipeline with a hyperparameter optimization step using RandomizedSearchCV
        pipeline_random_forest_simple = Pipeline(
            steps=[ (  
                    "hyper_opt",                           
                    RandomizedSearchCV(                    
                        RandomForestClassifier(class_weight='balanced'),          
                        param_distributions=param_dist,    
                        n_iter=n_iter,                     # Number of hyperparameters combination sampled
                        cv=cv,                             # Cross-validation folds 
                        n_jobs=-1                          # Use all available cores for parallel processing                  
                    ),
                )
            ]
        )

        logging.info("Fitting Random Forest pipeline with RandomizedSearchCV.")
        pipeline_random_forest_simple.fit(X_tr, y_tr)

        # Predict labels and probabilities for the test set
        y_pred = pipeline_random_forest_simple.predict(X_tst)
        y_prob = pipeline_random_forest_simple.predict_proba(X_tst)[:, 1]
        
        
        # Evaluate model performance
        metrics_scores = evaluate_model_performance(y_tst, y_pred,y_prob)
        
        model = pipeline_random_forest_simple.named_steps['hyper_opt'].best_estimator_

        
        metrics_logger.append_metrics(metrics_scores, y_tst, y_prob, model)
        
        

        

    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(metrics_logger.tpr_list, metrics_logger.auc_list)
    
    # Plot mean ROC curve
    plot_roc_curve(mean_fpr, mean_tpr, mean_auc, mean_auc_err)    #PROBLEMA

    # Calculate mean and standard deviation for each metric's list 
    (
        mean_acc,  # Mean accuracy
        mean_prec,  # Mean precision
        mean_rec,   # Mean recall
        mean_f1,    # Mean F1-score
        mean_spec,  # Mean specificity
        err_acc,    # Standard error for accuracy
        err_prec,   # Standard error for precision
        err_rec,    # Standard error for recall
        err_f1,     # Standard error for F1-score
        err_spec    # Standard error for specificity
    ) = compute_mean_std_metric(metrics_logger.accuracy_list, metrics_logger.precision_list, metrics_logger.recall_list, metrics_logger.f1_list, metrics_logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    metrics_logger.model_auc_pairs.sort(key=lambda x: x[1])
    pipeline_medianAUC_random_forest_simple = metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][0]

    # Log information about the selected model
    logging.info(
        f"The model with the median AUC (value: {metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )

    # Return the trained pipeline
    return pipeline_medianAUC_random_forest_simple



def RFPipeline_PCA(df1, df2, n_iter, cv):
    """
    
    Train and evaluate a Random Forest classifier pipeline with PCA for dimensionality reduction.
    
    The function performs 10 iterations of training and evaluation:
    
        - Converts input DataFrames into numpy arrays and maps categorical labels to binary (0, 1).
        - Defines a hyperparameter search space for the Random Forest classifier.
        - Splits data into train/test sets (90%/10%) for each iteration.
        - Defines a pipeline applying PCA followed by RandomizedSearchCV for hyperparameter tuning.
        - Trains the pipeline on the training data.
        - Predicts on the test set and computes predicted probabilities.
        - Computes performance metrics (accuracy, precision, recall, F1, specificity).
        - Computes ROC curve and AUC.
        - Logs the number of PCA components used.
        - Aggregates results and plots mean ROC and performance metrics.
        - Returns the model from the iteration with median AUC.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Feature matrix containing independent variables.
    df2 : pandas.DataFrame
        Target labels corresponding to df1, with categorical classes (e.g., 'Normal', 'AD').
    n_iter : int
        Number of hyperparameter combinations sampled in RandomizedSearchCV.
    cv : int
        Number of cross-validation folds used during hyperparameter optimization.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        The trained pipeline including PCA and Random Forest classifier from the iteration with median AUC.
    
    Notes
    -----
    - PCA reduces dimensionality before classification, which can improve model performance on high-dimensional data.
    - Labels are mapped to binary format: 'Normal' -> 0, 'AD' -> 1.
    - RandomForestClassifier uses balanced class weights to handle class imbalance.
    - The function performs multiple iterations of training and evaluation to reduce variance in performance estimation.
    - Uses all available CPU cores for parallel processing during hyperparameter search.
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    
    """
    


    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    region = list(df1.columns.values)

    
    mean_fpr = np.linspace(0, 1, 100) 
    metrics_logger = MetricsLogger(mean_fpr)

    for iteration in range(10):
        
        logging.info(f"Iteration {iteration + 1} - Splitting data into train and test sets.")
        
        # Split the data into training and test sets (10% test data)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1)

        # Define a pipeline with PCA and RandomizedSearchCV
        pipeline_random_forest_PCA = Pipeline(steps=[("dim_reduction", PCA()),
                                    ("hyper_opt", RandomizedSearchCV(RandomForestClassifier(class_weight='balanced'),
                                                                        param_distributions=param_dist,
                                                                        n_iter=n_iter,
                                                                        cv=cv,
                                                                        n_jobs=-1))
                                    ]
                                )
        logging.info("Fitting Random Forest pipeline with PCA and RandomizedSearchCV.")
        pipeline_random_forest_PCA.fit(X_tr, y_tr)

         # Predict labels and probabilities for the test set
        y_pred = pipeline_random_forest_PCA.predict(X_tst)
        y_prob = pipeline_random_forest_PCA.predict_proba(X_tst)[:, 1]

        # Evaluate model performance
        metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)

        # Log the number of PCA components
        n_components = pipeline_random_forest_PCA["dim_reduction"].n_components_
        logging.info(f"Number of PCA components used: {n_components}")
        
        model = pipeline_random_forest_PCA.named_steps['hyper_opt'].best_estimator_

        
        metrics_logger.append_metrics(metrics_scores, y_tst, y_prob, model)
        
        



    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(metrics_logger.tpr_list, metrics_logger.auc_list)
    
    # Plot mean ROC curve
    plot_roc_curve(mean_fpr, mean_tpr, mean_auc, mean_auc_err)    

    # Calculate mean and standard deviation for each metric's list 
    (
        mean_acc,  # Mean accuracy
        mean_prec,  # Mean precision
        mean_rec,   # Mean recall
        mean_f1,    # Mean F1-score
        mean_spec,  # Mean specificity
        err_acc,    # Standard error for accuracy
        err_prec,   # Standard error for precision
        err_rec,    # Standard error for recall
        err_f1,     # Standard error for F1-score
        err_spec    # Standard error for specificity
    ) = compute_mean_std_metric(metrics_logger.accuracy_list, metrics_logger.precision_list, metrics_logger.recall_list, metrics_logger.f1_list, metrics_logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    metrics_logger.model_auc_pairs.sort(key=lambda x: x[1])
    
    pipeline_medianAUC_random_forest_PCA = metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][0]

    # Log information about the selected model
    logging.info(
        f"The model with the median AUC (value: {metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )



    # Return the trained pipeline
    return pipeline_medianAUC_random_forest_PCA
    

def RFPipeline_RFECV(df1, df2, n_iter, cv):  
    """
    
    Train a Random Forest model using Recursive Feature Elimination with Cross-Validation (RFECV)
    combined with hyperparameter tuning via RandomizedSearchCV.
    
    The function performs the following steps over multiple iterations:
    
        - Maps categorical labels in df2 to binary values (0 and 1).
        - Recursively eliminates less important features using RFECV with recall as scoring metric.
        - Performs randomized hyperparameter search to optimize the Random Forest classifier on selected features.
        - Repeats the process across multiple train-test splits to assess model stability.
        - Logs and visualizes feature importance from the best estimator.
        - Computes performance metrics including accuracy, precision, recall, F1-score, specificity, and AUC.
        - Plots mean ROC curves with confidence intervals and bar charts of metrics.
        - Selects and returns the pipeline corresponding to the iteration with the median AUC score.
        - Visualizes the top 8 feature importances as a pie chart.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame containing the independent variables (features). Rows correspond to samples 
        and columns correspond to features.
    df2 : pandas.DataFrame
        DataFrame containing the dependent variable (target labels). Labels are categorical strings 
        such as 'Normal' and 'AD', mapped internally to 0 and 1.
    n_iter : int
        Number of iterations (samples) for the RandomizedSearchCV to sample hyperparameter combinations.
    cv : int
        Number of cross-validation folds used both in RFECV for feature selection and in RandomizedSearchCV 
        for hyperparameter tuning.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline consisting of:
          - The RFECV feature selector fit on training data,
          - The optimized Random Forest classifier fit on the selected features.
        The returned pipeline corresponds to the model from the iteration with the median AUC score.
    
    Raises
    ------
    ValueError
        If input DataFrames have mismatched indices or incompatible shapes.
    
    Notes
    -----
    - RFECV uses recall as the scoring metric and eliminates features in steps of 2, 
      retaining at least 20 features.
    - Random Forest classifier uses balanced class weights to handle class imbalance.
    - Median-AUC model selection reduces bias from single train-test splits.
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
   
    """


    # Convert features and labels to numpy arrays
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    roi_names = np.array(df1.columns.values)
    
    


    mean_fpr = np.linspace(0, 1, 100) 
    metrics_logger = MetricsLogger(mean_fpr)


    for iteration in range(10):  
        logging.info(f"Iteration {iteration + 1} - Splitting data into train and test sets.")

        # Split data into training and test sets (10% test data)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1)

        # Perform RFECV for feature selection
        rf_base = RandomForestClassifier(n_estimators=200, class_weight='balanced')
        feature_selector = RFECV(estimator=rf_base, step=2, cv=cv, scoring='recall', min_features_to_select=20) 
        feature_selector.fit(X_tr, y_tr)

        # Get selected features and transform datasets
        selected_mask = feature_selector.get_support() 
        selected_ROIs = roi_names[selected_mask]
        X_training_selected = feature_selector.transform(X_tr)
        X_test_selected = feature_selector.transform(X_tst)

        logging.info(f"Selected features: {selected_ROIs}")

        # Optimize Random Forest on selected features
        rf_selected_features = RandomizedSearchCV(
            RandomForestClassifier(class_weight='balanced'),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1
        )
        rf_selected_features.fit(X_training_selected, y_tr)

        # Predict and evaluate model performance
        y_pred = rf_selected_features.predict(X_test_selected)
        y_prob = rf_selected_features.predict_proba(X_test_selected)
        metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)

        
        # Extract feature importances
        importances = rf_selected_features.best_estimator_.feature_importances_
        sorted_idx_iteration = np.argsort(importances)[::-1]  # Indices of features sorted by importance
        top2_ROIs = selected_ROIs[sorted_idx_iteration[:2]]  # Get the top 2 ROI names
        top2_importances = importances[sorted_idx_iteration[:2]]  # Get the top 2 importances
    
        logging.info(f"Top 2 important ROIs for iteration {iteration + 1}: {top2_ROIs} with importances {top2_importances}")



        # Create a pipeline with the feature selector and classifier
        pipeline_rfecv = Pipeline([
        ("selector", feature_selector),
        ("classifier", rf_selected_features.best_estimator_)
        ])

        metrics_logger.append_metrics(metrics_scores, y_tst, y_prob, pipeline_rfecv, selected_rois=selected_ROIs)



    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(metrics_logger.tpr_list, metrics_logger.auc_list)
    
    # Plot mean ROC curve
    plot_roc_curve(mean_fpr, mean_tpr, mean_auc, mean_auc_err)    

    # Calculate mean and standard deviation for each metric's list 
    (
        mean_acc,  # Mean accuracy
        mean_prec,  # Mean precision
        mean_rec,   # Mean recall
        mean_f1,    # Mean F1-score
        mean_spec,  # Mean specificity
        err_acc,    # Standard error for accuracy
        err_prec,   # Standard error for precision
        err_rec,    # Standard error for recall
        err_f1,     # Standard error for F1-score
        err_spec    # Standard error for specificity
    ) = compute_mean_std_metric(metrics_logger.accuracy_list, metrics_logger.precision_list, metrics_logger.recall_list, metrics_logger.f1_list, metrics_logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and select the one with the median value
    metrics_logger.model_auc_pairs.sort(key=lambda x: x[1])

    # Get index of model with median AUC
    median_index = len(metrics_logger.model_auc_pairs) // 2
    pipeline_rfecv_medianAUC = metrics_logger.model_auc_pairs[median_index][0]
    selector = pipeline_rfecv_medianAUC.named_steps['selector']
    selected_ROIs_median = roi_names[selector.get_support()]
    selected_ROIs_median = np.array(selected_ROIs_median)
    

    # Extract top 8 feature importances for selected ROIs
    importances = pipeline_rfecv_medianAUC.named_steps['classifier'].feature_importances_
    n_top = min(8, len(importances), len(selected_ROIs_median))
    sorted_idx = np.argsort(importances)[::-1][:n_top]
    top8_ROIs = selected_ROIs_median[sorted_idx]
    top8_importances = importances[sorted_idx]

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab20.colors  # IEEE-like palette
    patches, texts, autotexts = plt.pie(
        top8_importances,
        labels=top8_ROIs,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors[:n_top],
        textprops={'fontsize': 3, 'weight': 'bold'},
        radius=0.80
    )
    plt.title("Importance of the 8 best ROIs for the model with the median AUC", fontsize=6, weight='bold')
    plt.tight_layout()
    plt.show()

    logging.info(
        f"The model with the median AUC (value: {metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )


   
    return pipeline_rfecv_medianAUC
    




def SVM_simple(df1, df2, ker: str, cv: int):    
    """
    
    Train a Support Vector Machine (SVM) classifier with hyperparameter tuning via GridSearchCV.
    
    The function performs the following steps over multiple iterations:
    
        - Splits the input dataset into training and test sets.
        - Maps categorical target labels to binary integers (0 and 1).
        - Performs hyperparameter optimization on SVM parameters (kernel type, C, and gamma for RBF).
        - Enables probability estimates for AUC calculation.
        - Evaluates model performance using accuracy, precision, recall, F1-score, specificity, and AUC.
        - Repeats training and evaluation over multiple random splits to assess stability.
        - Selects and returns the SVM model pipeline with the median AUC score.
        - Produces visual outputs including ROC curves and bar charts of performance metrics.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame of independent variables (features) where rows are samples and columns are features.
    df2 : pandas.DataFrame
        DataFrame of dependent variable (target labels). Labels are categorical strings (e.g., 'Normal', 'AD') 
        mapped internally to binary integers {0, 1}.
    ker : str
        Kernel type for the SVM. Common choices are:
          - 'linear' : linear kernel,
          - 'rbf' : radial basis function (Gaussian) kernel.
        The choice affects which hyperparameters are tuned and model behavior.
    
    Returns
    -------
    sklearn.svm.SVC
        The best SVM model (fitted estimator) found by GridSearchCV, selected based on median AUC 
        performance across multiple iterations.
    
    Raises
    ------
    ValueError
        If an unsupported kernel type is provided or if input DataFrames have mismatched indices.
    
    Notes
    -----
    - For 'linear' kernel, hyperparameters tuned are `C` and class_weight.
    - For 'rbf' kernel, hyperparameters tuned include `C`, `gamma`, and class_weight.
    - Probability estimates are enabled for AUC calculation.
    - Class weights are balanced by default during fitting.
    - Model evaluation metrics include accuracy, precision, recall, F1-score, specificity, and AUC.
    - The median-AUC model is returned to ensure robustness over random splits.
    - Visual outputs include ROC curve plots and bar charts of performance metrics.
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
    """

    # Extract feature and target data as NumPy arrays
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values # convert labels to binary

    # Define the parameter grid for SVM
    if ker == 'linear':
        param_grid = {
                'C': np.logspace(-3, 3, 10),
                'kernel': [ker],
                'class_weight': ['balanced', None],
                'probability': [True]
            }
    else:
        param_grid = {
                'C': np.logspace(-3, 3, 10),
                'gamma': np.logspace(-4, 2, 10),
                'kernel': [ker],
                'class_weight': ['balanced', None],
                'probability': [True]
            }
            

  

    # Initialize metrics storage

    mean_fpr = np.linspace(0, 1, 100)
    metrics_logger = MetricsLogger(mean_fpr) 

    for iteration in range(10):
        logging.info(f"Iteration {iteration + 1}: splitting data into train/test sets.")

        # Split dataset
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1)

        # Initialize classifier and GridSearchCV
        classifier_svm = svm.SVC(kernel=ker, probability=True, class_weight='balanced')
    
        pipeline_SVM_grid_optimazed = GridSearchCV(
                    classifier_svm,
                    param_grid,
                    cv=cv,
                    scoring='roc_auc',
                    refit=True,
                    n_jobs=-1
                )
        # Train model
        pipeline_SVM_grid_optimazed.fit(X_tr, y_tr)
    
        # Predictions and probabilities
        y_pred = pipeline_SVM_grid_optimazed.predict(X_tst)
        y_prob = pipeline_SVM_grid_optimazed.predict_proba(X_tst)[:, 1]
    
        # Evaluate model performance 
        metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)
        
        metrics_logger.append_metrics(metrics_scores, y_tst, y_prob, pipeline_SVM_grid_optimazed.best_estimator_)
        

    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(metrics_logger.tpr_list, metrics_logger.auc_list)
    
    # Plot mean ROC curve
    plot_roc_curve(mean_fpr, mean_tpr, mean_auc, mean_auc_err)    

    # Calculate mean and standard deviation for each metric's list 
    (
        mean_acc,  # Mean accuracy
        mean_prec,  # Mean precision
        mean_rec,   # Mean recall
        mean_f1,    # Mean F1-score
        mean_spec,  # Mean specificity
        err_acc,    # Standard error for accuracy
        err_prec,   # Standard error for precision
        err_rec,    # Standard error for recall
        err_f1,     # Standard error for F1-score
        err_spec    # Standard error for specificity
    ) = compute_mean_std_metric(metrics_logger.accuracy_list, metrics_logger.precision_list, metrics_logger.recall_list, metrics_logger.f1_list, metrics_logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    metrics_logger.model_auc_pairs.sort(key=lambda x: x[1])
    
    pipeline_SVM_medianAUC = metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][0]

    # Log information about the selected model
    logging.info(
        f"The model with the median AUC (value: {metrics_logger.model_auc_pairs[len(metrics_logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )


    return pipeline_SVM_medianAUC


