# Standard library imports for file system operations
import sys
from pathlib import Path
import os

# Add the parent directory of the current working directory to the system path.
# This ensures that Python can locate modules stored in the parent directory. #MODIFIED
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
from scipy import stats
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFECV

# Adding a specific path to the system for importing custom modules
sys.path.append(r"C:\Users\daria\OneDrive\Desktop\NUOVO_GIT\CMEPDA-EXAM")  # #MODIFIED

# Importing a custom module for performance evaluation
import performance_scores

# Importing a custom module to interact with MATLAB Engine
import f_alternative_matlab_engine as feature_extractor  # #MODIFIED

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': stats.randint(50, 500),  # Number of trees in the forest
    'max_depth': stats.randint(1, 20)       # Maximum depth of the trees
}

# Function to create and train a Random Forest pipeline
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
    y = df2.loc[df1.index].values  # Align labels with features

    # Get column names to be used as feature names for visualization
    region = list(df1.columns.values)

    # Split data into training and test sets (10% test data)
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1, random_state=7)

    # Define a pipeline with a hyperparameter optimization step using RandomizedSearchCV
    pipeline_simple = Pipeline(
        steps=[
            (
                "hyper_opt",
                RandomizedSearchCV(
                    RandomForestClassifier(),          # Random Forest Classifier
                    param_distributions=param_dist,    # Parameter distributions for sampling
                    n_iter=n_iter,                     # Number of parameter settings to sample
                    cv=cv,                             # Cross-validation folds
                    random_state=10,                   # Seed for reproducibility
                ),
            )
        ]
    )

    # Train the pipeline on the training data
    pipeline_simple.fit(X_tr, y_tr)

    # Predict labels and probabilities for the test set
    y_pred = pipeline_simple.predict(X_tst)
    y_prob = pipeline_simple.predict_proba(X_tst)

    # Compute performance scores based on predictions
    scores = performance_scores(y_tst, y_pred, y_prob)

    # Save and visualize the first three decision trees in the forest
    best_estimator = pipeline_simple["hyper_opt"].best_estimator_
    for i, tree in enumerate(best_estimator.estimators_[:3]):  # Iterate over the first three trees
        dot_data = export_graphviz(
            tree,
            feature_names=region,  # Feature names
            filled=True,           # Fill colors for nodes
            impurity=False,        # Do not display impurity measures
            proportion=True,       # Show proportion of samples in each node
            class_names=["CN", "AD"],  # Class names for visualization
        )
        graph = graphviz.Source(dot_data)
        graph.render(view=True)  # Render the tree visualization

    # Return the trained pipeline
    return pipeline_simple  # #MODIFIED: Removed redundant `.fit` in return statement







