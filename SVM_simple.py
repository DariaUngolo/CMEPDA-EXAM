
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
import shutil

# 🔧 Aggiunge il percorso di Graphviz manualmente al PATH di sistema per Python
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# 🧪 (Opzionale) Debug: stampa dove si trova 'dot'
print("DOT trovato in:", shutil.which("dot"))

from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFECV
from scipy.stats import randint

# Adding a specific path to the system for importing custom modules
sys.path.append(r"C:\Users\daria\OneDrive\Desktop\CIAO\CMEPDA-EXAM")
#sys.path.append(r"C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM")


# Importing a custom module for performance evaluation
from performance_scores import compute_binomial_error, evaluate_model_performance

# Importing a custom module to interact with MATLAB Engine
import f_alternative_matlab_engine_NUOVOATLANTE as feature_extractor


param_dist = {
    'n_estimators': randint(50, 500),  # Numero di alberi
    'max_depth': randint(1, 20) }      # Profondità massima dell'albero
   # 'min_samples_split': randint(5, 15),  # Numero minimo di campioni per fare una divisione
   # 'min_samples_leaf': randint(1, 5),   # Numero minimo di campioni per foglia
   # 'max_features': ['sqrt', 'log2'],    # Tipo di features da considerare in ogni albero
   # 'bootstrap': [True, False]  }         # Attiva o disattiva il campionamento bootstrap
#}
#param_dist = {
#    "hyper_opt__n_estimators": randint(100, 600),    # Più alberi per stabilità
#    "hyper_opt__max_depth": randint(5, 40),         # Estendi la profondità massima
#    "hyper_opt__min_samples_split": randint(2, 20),
#    "hyper_opt__min_samples_leaf": randint(1, 10),
#    "hyper_opt__max_features": ['sqrt', 'log2', None],
#    "hyper_opt__bootstrap": [True, False]
#}



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
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values #converti le etichette in numeri


    if ker == 'linear':
        param_grid = {
                'C': np.logspace(-3, 3, 10),
                'gamma': np.logspace(-4, 2, 10),
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

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

    clf = svm.SVC(kernel=ker)

    grid = GridSearchCV(clf, param_grid, refit=True, scoring='roc_auc', cv=5)

    # fitting the model for grid search
    grid.fit(X_tr, y_tr)

    y_pred = grid.predict(X_tst)
    y_prob = grid.decision_function(X_tst)

    # SCORES WITH 95% CONFIDENCE INTERVAL
    scores = evaluate_model_performance(y_tst, y_pred, y_prob)

    return grid.pip