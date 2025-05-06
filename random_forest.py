import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))

"""

    Si basa sulla costruzione di una collezione di alberi decisionali, in cui ogni albero contribuisce a produrre una previsione finale
    Abbiamo un insieme di alberi decisionali, alberi indipendenti messi insieme formano il RandomForest
    L'algoritmo foresta casuale è costituito da una raccolta di decision trees e ogni albero dell'insieme è composto
    da un campione di dati tratto da un set di addestramento con sostituzione, chiamato Bootstrapping. Di quel campione di allenamento,
    un terzo viene accantonato come dati di test, noti come campione out-of-bag(oob), su cui torneremo più avanti.

"""

# Importazioni di librerie per il Machine Learning
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
from ML_tools.score_and_error import performance_scores



#devo importare la funzione di python
import feature_extractor


# PARAMETERS DISTRIBUTION FOR RANDOMSEARCH
# RandomForestClassifier: Modello di classificazione basato su una foresta di alberi decisionali
.
param_dist = {'n_estimators': stats.randint(50, 500),   #chat dice >300 non offre un miglioramento proporzionale in prestazioni, ma aumenta il costo computazionale
              'max_depth': stats.randint(1, 20)}        # chat dice [1,15]




#n_estimators rappresenta il numero di alberi nella foresta.
# Un valore maggiore generalmente migliora le prestazioni del modello, ma aumenta il costo computazionale.
# Per dataset piccoli o medi, 50< n estimators < 200 è spesso sufficiente
# Prova diversi valori per capire il punto in cui il miglioramento delle metriche diventa trascurabile (diminishing returns).
# max_depth controlla la profondità massima di ogni albero. Limitarla previene overfitting,
#Se Non Sei Sicuro:1 <max_depth<20.

# RANDOM FOREST

def RFPipeline_noPCA(df1, df2, n_iter, cv):
    """
    Creates pipeline that perform Random Forest classification on the data without Principal Component Analysis. The
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
    pipeline_simple : sklearn.pipeline.Pipeline
        A fitted pipeline (includes hyperparameter optimization using RandomizedSearchCV and a Random Forest Classifier
        model).

    See Also
    --------
    RandomizedSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    """

    X = df1.values

    df2 = df2.loc[df1.index]

    # df1 è un DataFrame Pandas che contiene i dati delle caratteristiche (feature).
    # Il metodo .values converte il DataFrame in un array NumPy.
    # X diventa una matrice in cui:
    # Ogni riga rappresenta un esempio. Ogni colonna rappresenta una caratteristica.


    y = df2.values
    # df2 è un DataFrame che contiene le etichette o target associati ai dati.



    region = list(df1.columns.values) #region = list(df1.columns.values)
    #Estrae i nomi delle colonne (le caratteristiche) da df1 e li converte in una lista

    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=7)
    # Divide i dati in: Training set (X_tr, y_tr): Dati usati per addestrare il modello.
    # Test set (X_tst, y_tst): Dati usati per valutare il modello.
    # test_size=.1: Il 10% dei dati è destinato al test, mentre il restante 90% all'addestramento.
    # random_state=7: Garantisce che la suddivisione sia riproducibile

    pipeline_simple = Pipeline(steps=[("hyper_opt", RandomizedSearchCV(RandomForestClassifier(),        #Una pipeline   combina diversi passaggi in un flusso
                                                                                                        #sequenziale, rendendo il processo più organizzato.
                                                                       param_distributions=param_dist,
                                                                       n_iter=n_iter,
                                                                       cv=cv,
                                                                       random_state=10))
                                      ]
                               )
    #RandomizedSearchCV:
    #Esegue una ricerca casuale sui parametri iper del modello per trovare la combinazione migliore.
    #RandomForestClassifier(): Il modello di classificazione scelto
    #n_iter: Numero di combinazioni di parametri da testare.
    #cv: Numero di fold nella validazione incrociata (cross-validation)
    #random_state=10: Garantisce risultati riproducibili.





    pipeline_simple.fit(X_tr, y_tr)   #pipeline_simple.fit(X_tr, y_tr)
                                      #Addestra la pipeline sui dati di training (X_tr, y_tr).

    y_pred = pipeline_simple.predict(X_tst)    #Genera le previsioni sui dati di test (X_tst) usando il modello     ottimizzato, y_pred contiene le etichette previste.
    y_prob = pipeline_simple.predict_proba(X_tst)
    #y_prob è una matrice in cui:
        # Ogni riga rappresenta un esempio.
        #Ogni colonna rappresenta la probabilità associata a una classe

    # SCORES WITH 95% CONFIDENCE INTERVAL
    scores = performance_scores(y_tst, y_pred, y_prob)  #Questa riga calcola i punteggi di prestazione del
    #modello confrontando le etichette reali (y_tst) con quelle predette (y_pred) e le probabilità predette (y_prob)

    for i in range(3):
        tree = pipeline_simple["hyper_opt"].best_estimator_[i]
        #pipeline_simple: Un pipeline di machine learning che contiene il modello addestrato e i passi precedenti
        # ["hyper_opt"]: Accede al passo chiamato "hyper_opt", che è un'istanza di RandomizedSearchCV.
        #Questo oggetto ha selezionato il miglior modello dopo aver esplorato diversi set di iperparametri.
        #i]: Seleziona il i-esimo albero di decisione dalla foresta casuale





        # Conversione dell'albero in formato grafico DOT
        dot_data = export_graphviz(tree,
                                   feature_names=region,
                                   filled=True,
                                   impurity=False,
                                   proportion=True,
                                   class_names=["CN", "AD"])
        graph = graphviz.Source(dot_data)

    #Visualizzare e analizzare: I primi i alberi di decisione della foresta casuale, per capire come vengono prese le decisioni.

        graph.render(view=True)

    return pipeline_simple.fit


mean_md, std_md, group = feature_extractor(paths_MD, paths_masks)
classifiers.RFPipeline_noPCA(mean_fa, group, 5, 5)



    """

    DATI INPUT CHE DOBBIAMO USARE  per eseguirlo

        df1:

            Un DataFrame contenente le caratteristiche (features) per il modello.
            Ogni riga rappresenta un esempio (ad esempio, un soggetto).
            Ogni colonna rappresenta una caratteristica (ad esempio, i valori medi per una ROI)



                        ROI_1  ROI_2  ROI_3  ROI_4
            Subject_1    0.8    0.5    0.4    0.3
            Subject_2    0.6    0.7    0.8    0.9
            Subject_3    0.9    0.6    0.7    0.8



        df2:

            Un DataFrame contenente le etichette (target) corrispondenti agli esempi in df1.
            Dovrebbe avere una singola colonna o una struttura equivalente.

                        Label
            Subject_1      0
            Subject_2      1
            Subject_3      0


        n_iter:

            Numero di combinazioni casuali di parametri da esplorare.
            Esempio: n_iter = 10


        cv:

            Numero di fold da usare nella validazione incrociata.
            Esempio: cv = 3

    """