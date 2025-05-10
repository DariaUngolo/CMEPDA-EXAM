import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def compute_binomial_error(metric_value, n_samples, confidence_level):
    """
    Estimate the error of a binomial metric (accuracy, precision, recall)
    using normal approximation.

    Parameters:
    -----------
    metric_value : float
        The metric value (between 0 and 1), such as accuracy, precision, or recall.

    n_samples : int
        Number of independent observations (e.g., test set size).

    confidence_level : float
        Confidence level (e.g., 0.683 for 68.3% CI, 0.95 for 95% CI).

    Returns:
    --------
    float
        Estimated error based on the z-score and binomial variance.
    
    Notes:
    ------
    The error is computed using the normal approximation of the binomial distribution:
        error ≈ z * sqrt(p * (1 - p) / n)
    where:
        - z is the z-score for the specified confidence level,
        - p is the metric value,
        - n is the number of samples.

    This approximation assumes sufficiently large n (n * p and n * (1 - p) ≥ ~5).
    """
    z = norm.ppf((1 + confidence_level) / 2.0)
    return z * np.sqrt((metric_value * (1 - metric_value)) / n_samples)


def evaluate_model_performance(y_true, y_pred, y_proba, confidence_level=0.683):
    """
    Evaluate binary classification performance with metric errors and ROC visualization.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).

    y_pred : array-like of shape (n_samples,)
        Predicted labels by the classifier.

    y_proba : array-like of shape (n_samples,) or (n_samples, 2)
        Predicted class probabilities. If 2D, second column should correspond to class 1.

    confidence_level : float, optional (default=0.683)
        Desired confidence level for error estimation (e.g., 0.683 ≈ ±1σ, 0.95 ≈ ±2σ).

    Returns:
    --------
    dict
        Dictionary with performance metrics and their estimated errors:
        {
            'Accuracy': value, 'Accuracy_error': err,
            'Precision': value, 'Precision_error': err,
            'Recall': value, 'Recall_error': err,
            'AUC': value, 'AUC_error': err
        }
    """

    # Ensure probabilities are 1D for the positive class
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    # Compute core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Estimate errors
    n = len(y_true)
    acc_err = compute_binomial_error(accuracy, n, confidence_level)
    prec_err = compute_binomial_error(precision, n, confidence_level)
    rec_err = compute_binomial_error(recall, n, confidence_level)

    # ROC and AUC computation
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label= 1)
    roc_auc = auc(fpr, tpr)

    # AUC error estimation based on classical Hanley & McNeil (1982) approach
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc ** 2 / (1 + roc_auc)

    z = norm.ppf((1 + confidence_level) / 2.0)
    auc_err = z * np.sqrt(
        (roc_auc * (1 - roc_auc) +
         (n1 - 1) * (q1 - roc_auc ** 2) +
         (n0 - 1) * (q2 - roc_auc ** 2)) / (n1 * n0)
    )

    # Plot ROC Curve (aesthetically improved)
    #plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5,
             label=f'ROC curve (AUC = {roc_auc:.2f} ± {auc_err:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Bar Plot with Error Bars ---
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    values = [accuracy, precision, recall, roc_auc]
    errors = [acc_err, prec_err, rec_err, auc_err]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, values, yerr=errors, capsize=8, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'],
                  edgecolor='black', linewidth=1.2)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics with Confidence Errors", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Accuracy:  {accuracy:.2f} ± {acc_err:.2f}")
    print(f"Precision: {precision:.2f} ± {prec_err:.2f}")
    print(f"Recall:    {recall:.2f} ± {rec_err:.2f}")
    print(f"AUC:       {roc_auc:.2f} ± {auc_err:.2f}")

    return {
        'Accuracy': accuracy, 'Accuracy_error': acc_err,
        'Precision': precision, 'Precision_error': prec_err,
        'Recall': recall, 'Recall_error': rec_err,
        'AUC': roc_auc, 'AUC_error': auc_err
    }




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