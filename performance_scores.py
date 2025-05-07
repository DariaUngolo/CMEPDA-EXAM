import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def performance_scores(y_test, y_predicted, y_probability, confidence_int=0.683):
    """
    Calculate various performance metrics for a binary classification model along with their associated errors.

    Parameters:
    -----------
    y_test : array-like
        The true labels of the test set (a binary label array).

    y_predicted : array-like
        The predicted labels by the model (same shape as y_test).

    y_probability : array-like
        The predicted probabilities for each class (model's prediction confidence).

    confidence_int : float, optional, default=0.683
        The confidence interval for calculating the metric errors (default corresponds to approximately one standard deviation, i.e., 68.3%).

    Returns:
    --------
    dict
        A dictionary containing the calculated metrics and their errors:
        'Accuracy', 'Precision', 'Recall', 'AUC' along with their respective errors.
    """

    # If y_probability is a 2D matrix (e.g., probabilities for both classes 0 and 1), select the probability for the positive class (class 1)
    if y_probability.ndim == 2:
        y_prob = y_probability[:, 1]  # Selecting the probability for class 1 (positive class) #CAMBIATO
    else:
        y_prob = y_probability  # Otherwise, y_probability is already a 1D array for the positive class

    # Calculate the z-score based on the desired confidence interval
    z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)  # z-score for confidence interval

    # Calculate the performance metrics
    precision = precision_score(y_test, y_predicted)  # Precision: True Positives / (True Positives + False Positives)
    accuracy = accuracy_score(y_test, y_predicted)  # Accuracy: Correct Predictions / Total Predictions
    recall = recall_score(y_test, y_predicted)  # Recall: True Positives / (True Positives + False Negatives)

    # Function to calculate the error associated with a metric
    def metric_error(metric, n):
        """
        Calculate the error for a given performance metric using the z-score and sample size.

        Parameters:
        -----------
        metric : float
            The value of the metric (accuracy, precision, recall).

        n : int
            The number of samples in the test set.

        Returns:
        --------
        float
            The error associated with the metric.
        """
        return z_score * np.sqrt((metric * (1 - metric)) /  sample_size)  # Error calculation formula

    # Calculate the errors for each metric

    accuracy_err = compute_error(accuracy, len(y_test))  # Error in accuracy
    precision_err = compute_error(precision, len(y_test))   # Error in precision
    recall_err = compute_error(recall, len(y_test))  # Error in recall


    # Compute the false positive rate (fpr) and true positive rate (tpr) for the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)  # Compute ROC curve

    # Calculate the Area Under the Curve (AUC) for the ROC curve
    roc_auc = auc(fpr, tpr)  # AUC is the area under the ROC curve

    # Calculate the number of samples in the positive and negative classes
    n1 = sum(y_test == 1)  # Number of positive samples
    n2 = sum(y_test == 0)  # Number of negative samples

    # Compute the variance for the positive and negative classes
    q1 = roc_auc / (2 - roc_auc)  # Variance for the positive class
    q2 = 2 * roc_auc ** 2 / (1 + roc_auc)  # Variance for the negative class

    # Calculate the error in the AUC
    auc_err = z_score * np.sqrt(
        (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
    )  # APPROCCIO CLASSICO CALCOLO ERRORE auc_err = z_score * np.sqrt(roc_auc * (1 - roc_auc) / n)

    # Plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])  # Set x-axis limits
    plt.ylim([0.0, 1.05])  # Set y-axis limits
    plt.xlabel('False Positive Rate')  # Label for x-axis
    plt.ylabel('True Positive Rate')  # Label for y-axis
    plt.title('Receiver Operating Characteristic Curve')  # Title of the plot
    plt.legend(loc="lower right")  # Display the legend
    plt.show()  # Show the plot

    # Print the calculated metrics along with their errors
    print("Accuracy:", round(accuracy, 2), "+/-", round(accuracy_err, 2))  # Accuracy and its error
    print("Precision:", round(precision, 2), "+/-", round(precision_err, 2))  # Precision and its error
    print("Recall:", round(recall, 2), "+/-", round(recall_err, 2))  # Recall and its error
    print("AUC:", round(roc_auc, 2), "+/-", round(auc_err, 2))  # AUC and its error

    # Create a dictionary with all the metrics and their errors for easy access
    scores = {
        "Accuracy": accuracy, "Accuracy_error": accuracy_err,
        "Precision": precision, "Precision_error": precision_err,
        "Recall": recall, "Recall_error": recall_err,
        "AUC": roc_auc, "AUC_error": auc_err
    }

    return scores  # Return the dictionary of scores and errors





















#SPIEGAZIONI

#---- y_probability è una matrice a due dimensioni (ad esempio, se ci sono probabilità per entrambe le classi 0 e 1), seleziona le probabilità della classe positiva (classe 1

#---#Il punteggio Z calcolato in questo modo ti dice quanto lontano (in termini di deviazioni standard) devi andare dalla media per catturare una certa percentuale (intervallo di confidenza) dei dati.

   #Il punteggio Z è una misura che esprime quanti scarti tipici
    #(o deviazioni standard) un determinato valore si discosta dalla media
    #in una distribuzione normale

    #confidence_int: è l'intervallo di confidenza che si desidera
    #(1 + confidence_int) / 2.0: questa espressione calcola
    #il percentile superiore dell'intervallo di confidenza.

    #scipy.stats.norm.ppf(...) calcola il valore del punteggio Z corrispondente
    # alla probabilità specificata.




    #z_score = scipy.stats.norm.ppf((1 + 0.683) / 2.0)  # ppf(0.8415)
    #Il risultato di questa espressione sarà 1, perché nella distribuzione
    #normale standard il punteggio Z per un intervallo di confidenza del 68.3%
    #è 1 (circa). Questo significa che, in una distribuzione normale standard,
    # il 68.3% dei dati si trovano all'interno di ±1 deviazione standard dalla media.


    #accuratezza=num risposte corrette/totale
    #accuracy_err = z_score * np.sqrt((accuracy * (1 - accuracy)) / y_test.shape[0])
    #formula errore accuratezza sqrt((p(p+1))/N) dove p è la probabilità
    #che la risposta sia corretta


    #precisione = vero positivi/(falsi negativi e vero positivi)
    #la precisione misura la proporzione di predizioni positive corrette rispetto a tutte le predizioni positive:
    #precision_err = z_score * np.sqrt((precision * (1 - precision)) / y_test.shape[0])
    #formula errore precisione


    #Il richiamo misura la proporzione di predizioni positive corrette rispetto a tutte le etichette positive
    # richiamo: vero positivi/(vero positivi + falso negativi)
    #recall_err = z_score * np.sqrt((recall * (1 - recall)) / y_test.shape[0])


    #La funzione roc_curve calcola i tassi di:
    #False Positive Rate (FPR): La proporzione di negativi reali classificati erroneamente come positivi.
    #True Positive Rate (TPR): La proporzione di positivi reali classificati correttamente.
    #Questi valori vengono calcolati per diverse soglie di decisione sul punteggio di probabilità (y_prob).
   #pos_label=1 indica che la classe positiva è rappresentata dall'etichetta 1


    #La funzione auc calcola l'Area Under the Curve (AUC) della ROC curve.
    #L'AUC misura la capacità del modello di distinguere tra le classi: un valore di 1 rappresenta una separazione perfetta, mentre un valore di 0.5 indica una performance casuale



    """
    y_test= Gli etichette reali del set di test (un array di etichette binarie
    y_predicted=Le etichette predette dal modello
    y_probability: Le probabilità predette dal modello per ogni classe.
    In un problema di classificazione binaria
    confidence_int=(default è 0.683, corrispondente a un intervallo di confidenza
    di circa un sigma, ovvero il 68.3%)

    IL CODICE SE LO RUNNO NON MI DA PROBLEMI

    """

    """

    DOC STRING  VECCHIA

    Computes and displays various performance scores (including accuracy, precision, recall and AUC) with related errors
    for binary classification models.

    Parameters
    ----------
    y_test : numpy.ndarray
        True labels of test set.
    y_predicted : numpy.ndarray
        Predicted labels of test set.
    y_probability : numpy.ndarray
        Predicted label probabilities of test set.
    confidence_int : float, optional
        Confidence interval for error estimation. Default value is 0.683 (approximately 1 sigma).

    Returns
    -------
    scores : dict
        Dictionary containing various performance scores (and relative errors) including: Accuracy, Precision, Recall
        and AUC.

    See Also
    --------
    accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    roc_curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    auc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    """