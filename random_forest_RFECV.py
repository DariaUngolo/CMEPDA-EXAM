import matplotlib.pyplot as plt
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

from scipy import stats
from scipy.stats import randint
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
    'n_estimators': randint(100, 500),  # Numero di alberi
    'max_depth': randint(5, 50),        # Profondità massima dell'albero
    #'min_samples_split': randint(5, 15),  # Numero minimo di campioni per fare una divisione
    #'min_samples_leaf': randint(1, 5),   # Numero minimo di campioni per foglia
    #'max_features': ['sqrt', 'log2'],    # Tipo di features da considerare in ogni albero
    #'bootstrap': [True, False]           # Attiva o disattiva il campionamento bootstrap
}

def RFPipeline_RFECV_Top10ROI(df1, df2, n_iter, cv):
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values
    roi_names = np.array(df1.columns.values)

    # Split data
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1, random_state=7)

    # Feature selection (RFECV) solo sul training set
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    selector = RFECV(estimator=rf_base, step=2, cv=cv, scoring='recall', n_jobs=-1, min_features_to_select=20)
    selector.fit(X_tr, y_tr)

    # Maschera e nomi ROI selezionate
    selected_mask = selector.get_support()
    selected_ROIs = roi_names[selected_mask]

    # Riduci dataset
    X_tr_sel = selector.transform(X_tr)
    X_tst_sel = selector.transform(X_tst)

    print("ROI selezionate da RFECV:", selected_ROIs)

    # Ottimizzazione Random Forest su feature selezionate
    rf_clf = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced'),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        random_state=10,
        n_jobs=-1
    )
    rf_clf.fit(X_tr_sel, y_tr)

    # Predizione
    y_pred = rf_clf.predict(X_tst_sel)
    y_prob = rf_clf.predict_proba(X_tst_sel)

    # Score
    scores = evaluate_model_performance(y_tst, y_pred, y_prob)

    # Estrai importanza delle ROI selezionate
    best_rf = rf_clf.best_estimator_
    importances = best_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]  # Indici top 10 ROI
    top10_ROIs = selected_ROIs[sorted_idx]
    top10_importances = importances[sorted_idx]

    # Stampa il ranking completo delle ROI
    print("\nRanking completo delle ROI (importanza):")
    ranking_ROIs = sorted(zip(selected_ROIs, importances), key=lambda x: x[1], reverse=True)
    for rank, (roi, importance) in enumerate(ranking_ROIs, 1):
        print(f"{rank}. ROI: {roi}, Importanza: {importance:.4f}")

    # Plot a torta delle top 10 ROI
    plt.figure(figsize=(8, 6))
    plt.pie(top10_importances, labels=top10_ROIs, autopct='%1.1f%%', startangle=140)
    plt.title("Top 10 ROI (importanza Random Forest)")
    plt.axis('equal')
    plt.tight_layout()

    # Salva il grafico in formato PDF
    plt.savefig("top10_ROI_importance.pdf", format='pdf')

    plt.show()

    # Visualizza i primi 3 alberi
    for i, tree in enumerate(best_rf.estimators_[:3]):
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

    return rf_clf
