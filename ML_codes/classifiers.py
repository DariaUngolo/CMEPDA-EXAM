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




# Shared hyperparameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(5, 15),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}


def prepare_data(df_features, df_labels, test_size=0.1, random_state=None):
    X = df_features.values
    y = df_labels.loc[df_features.index].map({'Normal': 0, 'AD': 1}).values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_pipeline(pipeline, X_tr, X_tst, y_tr, y_tst, mean_fpr):
    """Fits the pipeline, evaluates metrics, and computes ROC/AUC."""
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_tst)
    y_prob = pipeline.predict_proba(X_tst)
    metrics_scores = evaluate_model_performance(y_tst, y_pred, y_prob)

    fpr, tpr, roc_auc, auc_err = compute_roc_and_auc(y_tst, y_prob)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    return metrics_scores, roc_auc, interp_tpr


def train_pipeline(
    X, y, pipeline_steps, n_iter, cv, iterations=2, test_size=0.1
):
    """Generic training function for machine learning pipelines."""
    mean_fpr = np.linspace(0, 1, 100)
    metrics = {
        'accuracy_list': [],
        'precision_list': [],
        'recall_list': [],
        'f1_list': [],
        'specificity_list': [],
        'auc_list': [],
        'tpr_list': []
    }
    model_auc_pairs = []

    for iteration in range(iterations):
        logging.info(f"Iteration {iteration + 1} - Splitting data.")
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=test_size)

        pipeline = Pipeline(pipeline_steps)
        metrics_scores, roc_auc, interp_tpr = evaluate_pipeline(
            pipeline, X_tr, X_tst, y_tr, y_tst, mean_fpr
        )

        for key, value in metrics_scores.items():
            metrics[f'{key.lower()}_list'].append(value)
        metrics['auc_list'].append(roc_auc)
        metrics['tpr_list'].append(interp_tpr)
        model_auc_pairs.append((pipeline, roc_auc))

    mean_tpr, mean_auc, mean_auc_err = compute_average_auc(
        metrics['tpr_list'], metrics['auc_list']
    )

    plot_roc_curve(mean_fpr, mean_tpr, mean_auc, mean_auc_err)
    performance_metrics = compute_mean_std_metric(
        metrics['accuracy_list'], metrics['precision_list'],
        metrics['recall_list'], metrics['f1_list'], metrics['specificity_list']
    )
    plot_performance_bar_chart(*performance_metrics, mean_auc_err)

    model_auc_pairs.sort(key=lambda x: x[1])
    best_model = model_auc_pairs[len(model_auc_pairs) // 2][0]
    logging.info(f"Selected model with median AUC.")
    return best_model


def RFPipeline_noPCA(df1, df2, n_iter, cv):
    X, y = df1.values, df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    steps = [("hyper_opt", RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced'), param_dist, n_iter=n_iter, cv=cv, n_jobs=-1
    ))]
    return train_pipeline(X, y, steps, n_iter, cv)


def RFPipeline_PCA(df1, df2, n_iter, cv):
    X, y = df1.values, df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    steps = [
        ("dim_reduction", PCA()),
        ("hyper_opt", RandomizedSearchCV(
            RandomForestClassifier(class_weight='balanced'), param_dist, n_iter=n_iter, cv=cv, n_jobs=-1
        ))
    ]
    return train_pipeline(X, y, steps, n_iter, cv)


def SVM_simple(df1, df2, ker, cv):
    X, y = df1.values, df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    param_grid = {
        'C': np.logspace(-3, 3, 10),
        'kernel': [ker],
        'gamma': np.logspace(-4, 2, 10) if ker != 'linear' else None,
        'class_weight': ['balanced', None],
        'probability': [True]
    }
    steps = [("hyper_opt", GridSearchCV(
        svm.SVC(kernel=ker, probability=True), param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
    ))]
    return train_pipeline(X, y, steps, n_iter=0, cv=cv)
