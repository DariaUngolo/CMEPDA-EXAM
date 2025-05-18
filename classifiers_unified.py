import sys
from pathlib import Path
import os

# Add the parent directory of the current working directory to the system path.
sys.path.insert(0, str(Path(os.getcwd()).parent))

"""
    Random Forest algorithm overview:
    It constructs a collection of decision trees, where each tree contributes to making a final prediction.
    The algorithm creates independent trees using a subset of training data (Bootstrapping). One-third of
    this subset is reserved as test data, known as out-of-bag (oob) samples, which are used to estimate the model’s performance.
"""

# Import essential libraries for Machine Learning
import numpy as np
import graphviz

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

from performance_scores import compute_binomial_error, evaluate_model_performance # Importing a custom module for performance evaluation

import feature_extractor # Importing a custom module to interact with MATLAB Engine

param_dist = {
    'n_estimators': randint(50, 500),  # number of trees
    'max_depth': randint(5, 50),        # maximum depth of the tree
    'min_samples_split': randint(5, 15),  # minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 5),   # minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'],    # number of features to consider when looking for the best split
    'bootstrap': [True, False]           # whether bootstrap samples are used when building trees
}


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

    
    # Extract feature and target data as NumPy arrays
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values # convert labels to binary


    # Get column names to be used as feature names for visualization
    region = list(df1.columns.values)
    
    # Split data into training and test sets (10% test data)
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1, random_state=7)

    # Define a pipeline with a hyperparameter optimization step using RandomizedSearchCV
    pipeline_random_forest_simple = Pipeline(
        steps=[ (  
                "hyper_opt",                           #"hyper_opt" is the name of the step in the pipeline
                RandomizedSearchCV(                    
                    RandomForestClassifier(class_weight='balanced'),          
                    param_distributions=param_dist,    
                    n_iter=n_iter,                     # Number of hyperparameters combination sampled
                    cv=cv,                             # Cross-validation folds
                    random_state=10,                   
                ),
            )
        ]
    )

    # Train the pipeline on the training data
    pipeline_random_forest_simple.fit(X_tr, y_tr) 

    # Predict labels and probabilities for the test set
    y_pred = pipeline_random_forest_simple.predict(X_tst)

    y_prob = pipeline_random_forest_simple.predict_proba(X_tst)

    # Compute performance scores based on predictions
    metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)
   

    # Save and visualize the first three decision trees in the forest
    best_estimator = pipeline_random_forest_simple["hyper_opt"].best_estimator_

    print(best_estimator.classes_) # Print the class labels of the best estimator

    for i, tree in enumerate(best_estimator.estimators_[:1]): 

        dot_data = export_graphviz(
    
            tree,                  
            feature_names=region, 
            
            filled=True,          # Fill nodes with color based on the class they predict
            impurity=False,        # Do not display impurity measures
            proportion=True,       # Display proportions of samples in each node
            class_names=["CN", "AD"],  
        )


        graph = graphviz.Source(dot_data) # Convert the dot data to a graph
        graph.render(view=True)  

    # Return the trained pipeline
    return pipeline_random_forest_simple


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

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    
    pipeline_random_forest_PCA = Pipeline(steps=[("dim_reduction", PCA()),
                                   ("hyper_opt", RandomizedSearchCV(RandomForestClassifier(),
                                                                    param_distributions=param_dist,
                                                                    n_iter=n_iter,
                                                                    cv=cv,
                                                                    random_state=10))
                                   ]
                            )

    pipeline_random_forest_PCA.fit(X_tr, y_tr)

    y_pred = pipeline_random_forest_PCA.predict(X_tst)
    y_prob = pipeline_random_forest_PCA.predict_proba(X_tst)

    metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)

    print("Components shape is:", np.shape(pipeline_random_forest_PCA["dim_reduction"].components_)[0])

     # Save and visualize the first decision tree in the forest
    best_estimator = pipeline_random_forest_PCA["hyper_opt"].best_estimator_

    print(best_estimator.classes_) # Print the class labels of the best estimator

    for i, tree in enumerate(best_estimator.estimators_[:1]): 

        dot_data = export_graphviz(
    
            tree,                  
            feature_names=region, 
            
            filled=True,          # Fill nodes with color based on the class they predict
            impurity=False,        # Do not display impurity measures
            proportion=True,       # Display proportions of samples in each node
            class_names=["CN", "AD"],  
        )


        graph = graphviz.Source(dot_data) # Convert the dot data to a graph
        graph.render(view=True)

    return pipeline_random_forest_PCA
    

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

    

    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    roi_names = np.array(df1.columns.values)

    # Split data into training and test sets (10% test data)
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1, random_state=7)

    # feature selection (RFECV) only on the training set to avoid data leakage
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    feature_selector = RFECV(estimator=rf_base, step=2, cv=cv, scoring='recall', n_jobs=-1, min_features_to_select=20) 
    feature_selector.fit(X_tr, y_tr)

    # Mask and selected ROI names
    selected_mask = feature_selector.get_support() # Get the mask of selected features
    selected_ROIs = roi_names[selected_mask]

    # reduce dataset to selected features
    X_training_selected = feature_selector.transform(X_tr)
    X_test_selected = feature_selector.transform(X_tst)

    print("ROI selezionate da RFECV:", selected_ROIs)

    # Random Forest optimization on selected features
    rf_selected_features = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced'),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        random_state=10,
        n_jobs=-1
    )
    rf_selected_features.fit(X_training_selected, y_tr)

    # prediction
    y_pred = rf_selected_features.predict(X_test_selected)
    y_prob = rf_selected_features.predict_proba(X_test_selected)

    # Scores
    metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)

    # extract feature importance of selected features
    best_rf = rf_selected_features.best_estimator_
    importances = best_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]  # indices of top 10 ROIs
    top10_ROIs = selected_ROIs[sorted_idx]
    top10_importances = importances[sorted_idx]

    # print the complete ranking of ROIs
    print("\nRanking completo delle ROI (importanza):")
    ranking_ROIs = sorted(zip(selected_ROIs, importances), key=lambda x: x[1], reverse=True)
    for rank, (roi, importance) in enumerate(ranking_ROIs, 1):
        print(f"{rank}. ROI: {roi}, Importanza: {importance:.4f}")

    # visualize best tree
    for i, tree in enumerate(best_rf.estimators_[:1]):
        dot_data = export_graphviz(
            tree,
            feature_names=selected_ROIs,
            filled=True,
            impurity=False,
            proportion=True,
            class_names=["CN", "AD"]
        )
        graph = graphviz.Source(dot_data)
        graph.render(view=True)

    return rf_selected_features


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
    # The parameters are defined based on the kernel type
    # If the kernel is 'linear', we don't need to specify 'gamma'
    # If the kernel is 'rbf', we need to specify 'gamma'
    # The 'class_weight' parameter is set to 'balanced' to handle class imbalance
    if ker == 'linear':
        param_grid = {
                'C': np.logspace(-3, 3, 10),
                'kernel': [ker],
                'class_weight': ['balanced', None]
            }
    else:
        param_grid = {
                'C': np.logspace(-3, 3, 10),
                'gamma': np.logspace(-4, 2, 10),
                'kernel': [ker],
                'class_weight': ['balanced', None]
            }

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6) # 10% test data

    # Create SVM classifier
    # The kernel is specified based on the input parameter
    classifier_svm = svm.SVC(kernel=ker)

    grid_optimized = GridSearchCV(classifier_svm, param_grid, refit=True, scoring='roc_auc', cv=5)

    # fitting the model for grid search
    grid_optimized.fit(X_tr, y_tr)

    y_pred = grid_optimized.predict(X_tst)
    y_prob = grid_optimized.decision_function(X_tst)

    # Compute performance scores based on predictions
    metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)

    return grid_optimized




