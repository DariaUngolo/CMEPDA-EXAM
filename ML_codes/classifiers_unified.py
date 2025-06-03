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



class MetricsLogger() :
    """
    A class to log and compute metrics for model evaluation.

    Attributes:
    -----------
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
    """
    def __init__(self, mean_fpr) :
    
        self.mean_fpr = mean_fpr
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.specificity_list = []
        self.auc_list = []
        self.tpr_list = []
        self.model_auc_pairs = []

    def append_metrics(self, metrics_scores, y_tst, y_prob, model) :
        """
        Append metrics, compute ROC/AUC, and store model information.

        Parameters:
        -----------
        metrics_scores : dict
            Dictionary containing evaluation metrics (e.g., accuracy, precision).

        y_tst : np.ndarray
            Ground truth labels for the test set.

        y_prob : np.ndarray
            Predicted probabilities for the test set.

        model : sklearn estimator
            The trained model to be logged.
        """
        self.accuracy_list.append(metrics_scores['Accuracy'])
        self.precision_list.append(metrics_scores['Precision'])
        self.recall_list.append(metrics_scores['Recall'])
        self.f1_list.append(metrics_scores['F1-score'])
        self.specificity_list.append(metrics_scores['Specificity'])
        
        fpr, tpr, roc_auc, auc_err = compute_roc_and_auc(y_tst, y_prob)
        self.auc_list.append(roc_auc)

        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tpr_list.append(interp_tpr)

        self.model_auc_pairs.append((model, roc_auc))

       


    

def RFPipeline_noPCA(df1, df2, n_iter, cv):
    """

    Train a Random Forest model pipeline without PCA.
    
    This function splits the dataset into training and test sets, performs hyperparameter optimization 
    using RandomizedSearchCV, and trains a Random Forest model. The trained pipeline is returned for 
    further evaluation or use.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Feature dataset containing independent variables.
    
    df2 : pandas.DataFrame
        Target dataset containing dependent variables (labels).
    
    n_iter : int
        Number of parameter combinations sampled during RandomizedSearchCV.
    
    cv : int
        Number of cross-validation folds used for hyperparameter tuning.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline with a trained Random Forest model and optimized hyperparameters.
    
    Notes
    -----
    - PCA is not applied in this pipeline.
    - The pipeline optimizes key parameters of the Random Forest classifier (e.g., number of trees, depth).
    - This method is suitable when dimensionality reduction is not required.
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    

    """

    
    # Convert features and labels to numpy arrays
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values # convert labels to binary


    # Get column names to be used as feature names for visualization
    region = list(df1.columns.values)
    

    mean_fpr = np.linspace(0, 1, 100) 
    logger = MetricsLogger(mean_fpr)

    for iteration in range(2):
        
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

        
        logger.append_metrics(metrics_scores, y_tst, y_prob, model)
        
        

        

    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(logger.tpr_list, logger.auc_list)
    
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
    ) = compute_mean_std_metric(logger.accuracy_list, logger.precision_list, logger.recall_list, logger.f1_list, logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    logger.model_auc_pairs.sort(key=lambda x: x[1])
    pipeline_medianAUC_random_forest_simple = logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][0]

    # Log information about the selected model
    logging.info(
        f"The model with the median AUC (value: {logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )

    # Return the trained pipeline
    return pipeline_medianAUC_random_forest_simple



def RFPipeline_PCA(df1, df2, n_iter, cv):
    """
    
    Train a Random Forest model pipeline with PCA.
    
    This function incorporates Principal Component Analysis (PCA) for dimensionality reduction 
    before training a Random Forest classifier. Hyperparameter optimization is performed using 
    RandomizedSearchCV.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Feature dataset containing independent variables.
    
    df2 : pandas.DataFrame
        Target dataset containing dependent variables (labels).
    
    n_iter : int
        Number of parameter combinations sampled during RandomizedSearchCV.
    
    cv : int
        Number of cross-validation folds used for hyperparameter tuning.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline that includes PCA and a trained Random Forest model.
    
    Notes
    -----
    - PCA reduces the feature space, which can improve model performance for high-dimensional datasets.
    - Optimal hyperparameters for the Random Forest are identified using RandomizedSearchCV.
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    """


    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    region = list(df1.columns.values)

    
    mean_fpr = np.linspace(0, 1, 100) 
    logger = MetricsLogger(mean_fpr)

    for iteration in range(2):
        
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

        
        logger.append_metrics(metrics_scores, y_tst, y_prob, model)
        
        



    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(logger.tpr_list, logger.auc_list)
    
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
    ) = compute_mean_std_metric(logger.accuracy_list, logger.precision_list, logger.recall_list, logger.f1_list, logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    logger.model_auc_pairs.sort(key=lambda x: x[1])
    
    pipeline_medianAUC_random_forest_PCA = logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][0]

    # Log information about the selected model
    logging.info(
        f"The model with the median AUC (value: {logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )



    # Return the trained pipeline
    return pipeline_medianAUC_random_forest_PCA
    

def RFPipeline_RFECV(df1, df2, n_iter, cv):  
    """

    Train a Random Forest model with recursive feature elimination and hyperparameter tuning.
    
    This function performs Recursive Feature Elimination with Cross-Validation (RFECV) to select the most important features.
    It then trains a Random Forest classifier using RandomizedSearchCV for hyperparameter optimization.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Feature dataset containing independent variables.
    
    df2 : pandas.DataFrame
        Target dataset containing dependent variables (labels).
    
    n_iter : int
        Number of hyperparameter combinations sampled during RandomizedSearchCV.
    
    cv : int
        Number of cross-validation folds for RFECV and hyperparameter search.
    
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model with optimized hyperparameters and selected features.
    
    Notes
    -----
    - RFECV recursively eliminates less important features to improve model performance.
    - Feature selection is done only on the training set to avoid data leakage.
    - The function ranks and prints feature importance, and saves a visualization of the best decision tree.
    
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
    logger = MetricsLogger(mean_fpr)


    for iteration in range(2):  
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
        sorted_idx = np.argsort(importances)[::-1]  # Indices of features sorted by importance
        top2_ROIs = selected_ROIs[sorted_idx[:2]]  # Get the top 2 ROI names
        top2_importances = importances[sorted_idx[:2]]  # Get the top 2 importances
    
        logging.info(f"Top 2 important ROIs for iteration {iteration + 1}: {top2_ROIs} with importances {top2_importances}")



        # Create a pipeline with the feature selector and classifier
        pipeline_rfecv = Pipeline([
        ("selector", feature_selector),
        ("classifier", rf_selected_features.best_estimator_)
        ])

        logger.append_metrics(metrics_scores, y_tst, y_prob, pipeline_rfecv)
        
        


    
    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(logger.tpr_list, logger.auc_list)
    
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
    ) = compute_mean_std_metric(logger.accuracy_list, logger.precision_list, logger.recall_list, logger.f1_list, logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    logger.model_auc_pairs.sort(key=lambda x: x[1])
    
    pipeline_rfecv_medianAUC = logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][0]

    # Extract feature importances for the selected features
    importances = pipeline_rfecv_medianAUC.named_steps['classifier'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:8]  # indices of top 10 ROIs
    n_top = min(8, len(selected_ROIs))
    sorted_idx = np.argsort(importances)[::-1][:n_top]
    top8_ROIs = selected_ROIs[sorted_idx]
    top8_importances = importances[sorted_idx]

        

    # Visualize feature importances with a pie chart
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab20.colors  # IEEE-like color palette
    patches, texts, autotexts = plt.pie(
            top8_importances,
            labels=top8_ROIs,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:8],
            textprops={'fontsize': 3, 'weight': 'bold'},
            radius=0.80,  # Adjust radius for better visibility
    )
    plt.title("Importance of the 8 best ROIs for the model with the median AUC", fontsize=6, weight='bold')
    plt.tight_layout()
        
    plt.show()
    # Log information about the selected model
    logging.success(
        f"The model with the median AUC (value: {logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )

   
    return pipeline_rfecv_medianAUC
    




def SVM_simple(df1, df2, ker: str):    
    """

    Train an SVM model pipeline with hyperparameter optimization.
    
    This function splits the dataset into training and test sets, performs hyperparameter optimization 
    using GridSearchCV, and trains a Support Vector Machine (SVM) model.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Feature dataset containing independent variables.
        
    df2 : pandas.DataFrame
        Target dataset containing dependent variables (labels).
        
    ker : str
        Kernel type for the SVM (e.g., 'linear', 'rbf').
    
    Returns
    -------
    sklearn.model_selection.GridSearchCV
        A fitted GridSearchCV object containing the best SVM model.
    
    Notes
    -----
    - The kernel type (`ker`) determines the decision boundary; 'linear' and 'rbf' are common choices.
    - GridSearchCV optimizes hyperparameters such as `C` (regularization) and `gamma` (for non-linear kernels).
    
    References
    ----------
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    
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
    logger = MetricsLogger(mean_fpr) 

    for iteration in range(2):
        logging.info(f"Iteration {iteration + 1}: splitting data into train/test sets.")

        # Split dataset
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1)

        # Initialize classifier and GridSearchCV
        classifier_svm = svm.SVC(kernel=ker, probability=True, class_weight='balanced')
    
        pipeline_SVM_grid_optimazed = GridSearchCV(
                    classifier_svm,
                    param_grid,
                    cv=20,
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
        
        logger.append_metrics(metrics_scores, y_tst, y_prob, pipeline_SVM_grid_optimazed.best_estimator_)
        

    # Calculate mean ROC curve and AUC with confidence intervals
    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(logger.tpr_list, logger.auc_list)
    
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
    ) = compute_mean_std_metric(logger.accuracy_list, logger.precision_list, logger.recall_list, logger.f1_list, logger.specificity_list)
        
        
    plot_performance_bar_chart(mean_acc, mean_prec, mean_rec, mean_f1, mean_spec, mean_auc, err_acc, err_prec,err_rec, err_f1, err_spec, mean_auc_err)

    # Sort models by AUC and return the one with median AUC
    logger.model_auc_pairs.sort(key=lambda x: x[1])
    
    pipeline_SVM_medianAUC = logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][0]

    # Log information about the selected model
    logging.info(
        f"The model with the median AUC (value: {logger.model_auc_pairs[len(logger.model_auc_pairs) // 2][1]:.3f}) "
        "has been selected and saved for further use."
    )


    return pipeline_SVM_medianAUC


