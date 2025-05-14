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
    'n_estimators': randint(50, 500),  # Numero di alberi
    'max_depth': randint(5, 50),        # Profondità massima dell'albero
    #'min_samples_split': randint(5, 15),  # Numero minimo di campioni per fare una divisione
    #'min_samples_leaf': randint(1, 5),   # Numero minimo di campioni per foglia
    #'max_features': ['sqrt', 'log2'],    # Tipo di features da considerare in ogni albero
    #'bootstrap': [True, False]           # Attiva o disattiva il campionamento bootstrap
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
    print("Indice di df1:")
    print(df1.index)
    print("Indice di df2:")
    print(df2.index)

    # Extract feature and target data as NumPy arrays
    X = df1.values
    y = df2.loc[df1.index].map({'Normal': 0, 'AD': 1}).values #converti le etichette in numeri


    # Get column names to be used as feature names for visualization
    region = list(df1.columns.values)
    #region è una lista che contiene i nomi delle colonne del tuo DataFrame, che corrispondono alle caratteristiche del tuo modello.

    # Split data into training and test sets (10% test data)
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1, random_state=7)

    # Define a pipeline with a hyperparameter optimization step using RandomizedSearchCV
    pipeline_simple = Pipeline(
        steps=[ (  #definisce il passaggio da fare, in questo caso abbiamo solo un passaggio,
        #Questa conterrà un solo passaggio, che è la ricerca dei migliori parametri per il modello di Random Forest.

                "hyper_opt",                           #"hyper_opt" è il nome dato a questo passaggio della pipeline.
                RandomizedSearchCV(                    # Algoritmo di ricerca per la selezione dei parametri
                    RandomForestClassifier(class_weight='balanced'),          # Random Forest Classifier. Questo è il nostro modello di Random Forest, che verrà ottimizzato.
                    param_distributions=param_dist,    # Parameter distributions for sampling, lo abbiamo definito fuori
                    n_iter=n_iter,                     # Number of parameter settings to sample  # Numero di combinazioni di parametri da testare, mmagina che tu abbia i seguenti parametri da ottimizzare per un Random Forest: Questo vuol dire che ci sono molte possibili combinazioni di valori.
                    #se imposti n_iter = 10, la ricerca esplorerà 10 combinazioni casuali di questi parametri.
                    cv=cv,                             # Cross-validation folds,  5 fold significa che i dati vengono suddivisi in 5 parti, e il modello viene allenato e testato 5 volte
                    random_state=10,                   # Seed for reproducibility. Quando imposti random_state=10, stai dicendo al tuo algoritmo di iniziare la sequenza di numeri casuali da un valore specifico (in questo caso, 10). Questo fa sì che la sequenza casuale generata sia sempre la stessa ogni volta che esegui il codice
                ),
            )
        ]
    )#in questo modello vogliamo ottimizzare il numero di alberi n_estimator e la profondità mac_depth
     #RandomizedSearchCV esplora casualmente diverse combinazioni di questi parametri e sceglie la combinazione che dà i migliori risultati

    # Train the pipeline on the training data
    pipeline_simple.fit(X_tr, y_tr)   #ALLENA I DATI USANDO I DATI DI TRAINNG

    # Predict labels and probabilities for the test set
    y_pred = pipeline_simple.predict(X_tst)
    #Questo comando utilizza il modello allenato
    #(contenuto nella pipeline pipeline_simple) per predire le etichette
    # delle osservazioni nel set di test X_tst, sono etichette discrete
    #Etichette di classe (es. [0, 1])
    y_prob = pipeline_simple.predict_proba(X_tst)#Questo comando restituisce le
     #probabilità associate a ciascuna classe per ogni osservazione nel set di test X_tst.                  Probabilità delle classi (es. [[0.8, 0.2], [0.3, 0.7]])
    # L'ordine delle colonne corrisponde all'ordine delle classi
    # nell'attributo classes_ del modello

    # Compute performance scores based on predictions
    scores = evaluate_model_performance(y_tst, y_pred, y_prob)
    #Calcola metriche di valutazione personalizzate tramite il modulo performance_scores
    #otteniamo il dizionario
    # return {
    #   'Accuracy': accuracy, 'Accuracy_error': acc_err,
    #    'Precision': precision, 'Precision_error': prec_err,
    #    'Recall': recall, 'Recall_error': rec_err,
    #    'AUC': roc_auc, 'AUC_error': auc_err
    #}


    # Save and visualize the first three decision trees in the forest
    best_estimator = pipeline_simple["hyper_opt"].best_estimator_
    # pipeline_simple["hyper_opt"]: Qui accedi al passo chiamato "hyper_opt" all'interno della pipeline.
            #best_estimator_: Questo attributo è disponibile dopo che RandomizedSearchCV ha terminato l'ottimizzazione dei parametri. best_estimator_ ti restituisce il modello con i migliori parametri trovati durante la ricerca. In questo caso, è il Random Forest con la combinazione ottimale di n_estimators, max_depth, ecc.
    # best_estimator è il miglior modello di Random Forest che RandomizedSearchCV ha trovato


    print(best_estimator.classes_) #per capire se ci da AD/CTRL nelle probabilità o CTRL/AD in y_prob = pipeline_simple.predict_proba(X_tst)

    for i, tree in enumerate(best_estimator.estimators_[:3]):  # Iterate over the first three trees(la ricerca casuale dei migliori parametri)
    #modello Random Forest è composto da una serie di alberi di decisione. L'attributo estimators_ contiene una lista di questi alberi. Ogni albero è un modello separato
    #[:3]: Qui stai selezionando i primi 3 alberi della foresta per visualizzarli.

        dot_data = export_graphviz(
    #export_graphviz è una funzione di scikit-learn che converte un albero di decisione in un formato grafico DOT, che è il formato utilizzato da Graphviz per la visualizzazione grafica degli alber
            tree,                  # L'albero che vuoi visualizzare
            feature_names=region,  # Feature names
            #region è una lista che contiene i nomi delle colonne del tuo DataFrame, che corrispondono alle caratteristiche del tuo modello.
            filled=True,           # Fill colors for nodes. Questo significa che i nodi dell'albero saranno colorati in base alla classe prevalente
            impurity=False,        # Do not display impurity measures
            proportion=True,       # filled=True:Se impostato su True, mostra la proporzione di campioni che arrivano in ciascun nod
            class_names=["CN", "AD"],  # Class names for visualization, I nomi delle classi per la classificazione
        )


        graph = graphviz.Source(dot_data)# graphviz.Source è utilizzato per creare un oggetto che rappresenta il grafico dell'albero decisionale.
        graph.render(view=True)  # Render the tree visualization,Questo comando renderizza l'albero (ovvero, crea l'immagine dell'albero decisionale) e apre automaticamente la visualizzazione

    # Return the trained pipeline
    return pipeline_simple.pip  # #MODIFIED: Removed redundant `.fit` in return statement







