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
    Creates and trains a Random Forest model pipeline without using PCA.

    This function splits the provided data into training and test sets, then performs hyperparameter optimization
    using RandomizedSearchCV to find the best Random Forest model configuration.
    The resulting pipeline is returned after training.

    Parameters:
        df1 (pandas.DataFrame): DataFrame containing the feature data (independent variables).
        df2 (pandas.DataFrame): DataFrame containing the target labels (dependent variable).
        n_iter (int): Number of parameter settings sampled during RandomizedSearchCV.
        cv (int): Number of cross-validation folds used during hyperparameter optimization.

    Returns:
        sklearn.pipeline.Pipeline: A fitted pipeline with a trained Random Forest classifier and optimized hyperparameters.

    Notes:
        - PCA is not used in this function.
        - The function optimizes the Random Forest classifier's parameters (number of trees, depth, etc.) using cross-validation.
        - This method is suitable for scenarios where dimensionality reduction is not required.

    --------
    RandomizedSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

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
        graph.render(view= False)  

    # Return the trained pipeline
    return pipeline_random_forest_simple


def RFPipeline_PCA(df1, df2, n_iter, cv):
    """
    Creates pipeline that perform Random Forest classification on the data with Principal Component Analysis. The
    input data is split into training and test sets, then a Randomized Search (with cross-validation) is performed to
    find the best hyperparameters for the model.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe containing the features.
    df2 : pandas.DataFrame
        Dataframe containing the labels.
    n_iter : int
        Number of parameter settings that are sampled.
    cv : int
        Number of cross-validation folds to use.

    Returns
    -------
    pipeline_PCA : sklearn.pipeline.Pipeline
        A fitted pipeline (includes PCA, hyperparameter optimization using RandomizedSearchCV and a Random Forest
        Classifier model).

    See Also
    --------
    PCA : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    RandomizedSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
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
        graph.render(view= False)

    return pipeline_random_forest_PCA
    

def RFPipeline_RFECV(df1, df2, n_iter, cv):
    """
        manca la documentazione
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
        graph.render(view= False) # Save the graph to a file without opening it

    return rf_selected_features


def SVM_simple(df1, df2, ker: str):
    """
    Performs SVM classification on the data. The input data is split into training and test sets, then a Grid Search
    (with cross-validation) is performed to find the best hyperparameters for the model. Feature reduction is not
    implemented in this function.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe containing the features.
    df2 : pandas.DataFrame
        Dataframe containing the labels.
    ker : str
        Kernel type.

    Returns
    -------
    grid : sklearn.model_selection.GridSearchCV
        A fitted grid search object with the best parameters for the SVM model.

    See Also
    --------
    GridSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
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


