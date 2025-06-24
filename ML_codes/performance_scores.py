import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_curve, auc, f1_score, confusion_matrix
)
from loguru import logger
import matplotlib
import seaborn as sns


def compute_binomial_error(metric_value, n_samples, confidence_level):
    """
    Estimate the error margin for a binomial metric using the normal approximation.

    This function calculates the error associated with metrics such as accuracy, precision,
    or recall by approximating the binomial distribution with a normal distribution.

    Parameters
    ----------
    metric_value : float
        The value of the metric (between 0 and 1), such as accuracy, precision, or recall.

    n_samples : int
        The number of independent observations (e.g., test set size).

    confidence_level : float
        The confidence level for the error estimation (e.g., 0.683 for 68.3% confidence, 0.95 for 95% confidence).

    Returns
    -------
    float
        The estimated error margin based on the z-score and binomial variance.

    Notes
    -----
    - This approximation assumes a sufficiently large number of samples (n), satisfying the condition: n * p and n * (1 - p) ≥ 5.
    - The z-score is derived from the standard normal distribution for the specified confidence level.

    Formula
    -------
    error ≈ z * sqrt(p * (1 - p) / n)

    """
    z = norm.ppf((1 + confidence_level) / 2.0)
    return z * np.sqrt((metric_value * (1 - metric_value)) / n_samples)

def bootstrap_confidence_interval(metric_func, y_true, y_pred, n_iterations=1000, confidence_level=0.683, random_state=42):
    """
    Estimate confidence interval of a metric via bootstrap resampling.

    Parameters
    ----------
    metric_func : callable
        Function that computes the metric (e.g., f1_score).
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    n_iterations : int
        Number of bootstrap resamples.
    confidence_level : float
        Confidence level (e.g., 0.683 = ±1σ).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated error margin (half-width of the confidence interval).
    """
    rng = np.random.default_rng(random_state)
    scores = []

    n = len(y_true)
    for _ in range(n_iterations):
        indices = rng.integers(0, n, size=n)
        score = metric_func(np.array(y_true)[indices], np.array(y_pred)[indices])
        scores.append(score)

    scores = np.sort(scores)
    lower_idx = int(((1 - confidence_level) / 2) * n_iterations)
    upper_idx = int((1 + confidence_level) / 2 * n_iterations)
    lower = scores[lower_idx]
    upper = scores[upper_idx]

    return (upper - lower) / 2  # half-width of confidence interval

def compute_roc_and_auc(y_true, y_prob):

    """
    Compute the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC).

    This function calculates the ROC curve and AUC metric for a binary classifier. Additionally, it estimates the
    error margin for the AUC using the Hanley & McNeil method.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    tuple
        A tuple containing false positive rates (fpr), true positive rates (tpr), AUC, and AUC error margin.

    """

    # Convert 2D array to 1D for the positive class
    confidence_level=0.683
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
        logger.debug("Converted 2D array of probabilities to 1D (positive class).")


    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)


    # Estimate AUC error using Hanley & McNeil method
    n1 = np.sum(y_true == 1)  # Number of positives
    n0 = np.sum(y_true == 0)  # Number of negatives
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc**2 / (1 + roc_auc)
    z = norm.ppf((1 + confidence_level) / 2.0)
    roc_auc_error = z * np.sqrt(
        (roc_auc * (1 - roc_auc) +
         (n1 - 1) * (q1 - roc_auc ** 2) +
         (n0 - 1) * (q2 - roc_auc ** 2)) / (n1 * n0)
    )

    logger.info("Computed AUC and related metrics: FPR, TPR, ROC AUC, and ROC AUC Error")




    return fpr, tpr, roc_auc, roc_auc_error


def plot_roc_curve(fpr, tpr, roc_auc, roc_auc_error):
    """

    Plot the Receiver Operating Characteristic (ROC) curve with AUC and error margin.

    Parameters
    ----------
    fpr : array-like
        False Positive Rates.

    tpr : array-like
        True Positive Rates.

    roc_auc : float
        Area Under the Curve (AUC) value.

    roc_auc_error : float
        Error margin for the AUC.

    Returns
    -------
    None

    """

    # Update matplotlib configuration for high-quality, compact plots
    matplotlib.rcParams.update({
        'font.size': 6,
        'font.family': 'serif',
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'legend.fontsize': 4,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'lines.linewidth': 0.8,
        'figure.dpi': 600,
        'savefig.dpi': 600
    })

    # Use colorblind-safe palette
    sns.set_palette("colorblind")

    logger.info("Generating ROC curve plot...")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(2, 1.4))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='tab:blue', label=f'AUC = {roc_auc:.2f} ± {roc_auc_error:.2f}', linewidth=0.8)
    
    # Diagonal line for random performance
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=0.6)

    # Axis labels and title
    ax.set_xlabel('False Positive Rate', labelpad=2, fontweight='semibold')
    ax.set_ylabel('True Positive Rate', labelpad=2, fontweight='semibold')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold', pad=4)
    ax.grid(axis='both', linestyle='--', linewidth=0.3, alpha=0.2)
    
    # Legend
    ax.legend(loc='lower right', frameon=False)

    # Improve layout and aesthetics
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(axis='both', direction='in', length=2, width=0.3)
    sns.despine(trim=True)

    plt.tight_layout()
    plt.show()

def compute_average_auc(tpr_list, auc_list):
    """
    Compute the average True Positive Rates (TPR) and AUC across multiple iterations.

    Parameters
    ----------
    tpr_list : list of arrays
        List of TPR arrays across multiple iterations.

    auc_list : list of floats
        List of AUC values across multiple iterations.

    Returns
    -------
    tuple
        Mean TPR, mean AUC, and error margin for the AUC.
    """
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    z = norm.ppf((1 + 0.683) / 2.0)
    auc_err = z * std_auc

    logger.success(f"Computed average AUC across all loops: {mean_auc:.4f} ± {auc_err:.4f}")

    return mean_tpr, mean_auc, auc_err



def evaluate_model_performance(y_true, y_pred, y_prob, confidence_level=0.683):
    """
    Evaluate the performance of a classification model using various metrics.

    This function computes common evaluation metrics such as accuracy, precision, recall,
    F1-score, specificity, and AUC. It also provides visualizations and estimates confidence
    intervals for each metric using binomial or Hanley-McNeil approximations.

    Parameters
    ----------
    y_true : array-like
        The ground truth (true labels).

    y_pred : array-like
        The predicted labels.

    y_prob : array-like
        The predicted probabilities for the positive class.

    confidence_level : float, optional
        Confidence level for error estimation (default: 0.683, equivalent to 1 standard deviation).

    Returns
    -------
    dict
        A dictionary containing the metric scores and their associated error estimates.

    """
    logger.info("Evaluating model performance...")

    # Ensure probabilities are 1D for the positive class
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
        logger.debug("Converted 2D array of probabilities to 1D (positive class).")

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division='warn')
    recall = recall_score(y_true, y_pred, zero_division='warn')
    f1 = f1_score(y_true, y_pred, zero_division='warn')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info("Computing error margins for classification metrics...")

    # Estimate errors using normal approximation
  
    accuracy_err = compute_binomial_error(accuracy, len(y_true), confidence_level)
    precision_err = compute_binomial_error(precision, tp+fp, confidence_level)
    recall_err = compute_binomial_error(recall, tp+fn, confidence_level)
    #f1_err = compute_binomial_error(f1, len(y_true), confidence_level)
    specificity_err = compute_binomial_error(specificity, tn+fp, confidence_level)

    # Bootstrap error for F1
    f1_err = bootstrap_confidence_interval(f1_score, y_true, y_pred, confidence_level=confidence_level)

    # Compute ROC curve and AUC
    fpr, tpr, roc_auc, auc_err = compute_roc_and_auc(y_true, y_prob)



    from tabulate import tabulate

    # Prepare table data
    table_data = [
        ["Accuracy",     f"{accuracy:.2f}",    f"±{accuracy_err:.2f}"],
        ["Precision",    f"{precision:.2f}",   f"±{precision_err:.2f}"],
        ["Recall",       f"{recall:.2f}",      f"±{recall_err:.2f}"],
        ["F1-score",     f"{f1:.2f}",          f"±{f1_err:.2f}"],
        ["Specificity",  f"{specificity:.2f}", f"±{specificity_err:.2f}"],
        ["AUC",          f"{roc_auc:.2f}",     f"±{auc_err:.2f}"]
    ]

    # Print formatted table
    print("\n" + tabulate(
        table_data,
        headers=["Metric", "Score", "± Error"],
        tablefmt="fancy_grid",
        colalign=("left", "center", "center")
    ))

    return {
        'Accuracy': accuracy, 'Accuracy_error': accuracy_err,
        'Precision': precision, 'Precision_error': precision_err,
        'Recall': recall, 'Recall_error': recall_err,
        'F1-score': f1, 'F1-score_error': f1_err,
        'Specificity': specificity, 'Specificity_error': specificity_err,
        'AUC': roc_auc, 'AUC_error': auc_err
    }

def compute_mean_std_metric(accuracy_list, precision_list, recall_list, f1_score_list, specificity_list):

    """

    Compute mean and standard deviation for classification metrics across multiple iterations.

    Parameters
    ----------
    accuracy_list : list of floats
        List of accuracy scores.

    precision_list : list of floats
        List of precision scores.

    recall_list : list of floats
        List of recall scores.

    f1_score_list : list of floats
        List of F1 scores.

    specificity_list : list of floats
        List of specificity scores.

    Returns
    -------
    tuple
        Mean values and standard errors for each metric.

    """

    # Calculate the mean and the standard deviation over all iteration
    mean_from_acc_list= np.mean(accuracy_list)
    err_from_acc_list = np.std(accuracy_list)

    mean_from_prec_list = np.mean(precision_list)
    err_from_prec_list = np.std(precision_list)

    mean_from_rec_list = np.mean(recall_list)
    err_from_rec_list = np.std(recall_list)

    mean_from_f1_list = np.mean(f1_score_list)
    err_from_f1_list = np.std(f1_score_list)

    mean_from_spec_list= np.mean(specificity_list)
    err_from_spec_list = np.std(specificity_list)



    # Return mean values and their errors
    logger.success("Computed mean values and their standard deviations for all classification metrics across loops.")

    from tabulate import tabulate

    table_title = "Average Classification Metrics Across Loops"

    # Prepare table data
    table_data = [
        ["Accuracy",     f"{mean_from_acc_list:.2f}",    f"±{err_from_acc_list:.2f}"],
        ["Precision",    f"{mean_from_prec_list:.2f}",   f"±{err_from_prec_list:.2f}"],
        ["Recall",       f"{mean_from_rec_list:.2f}",      f"±{err_from_rec_list:.2f}"],
        ["F1-score",     f"{mean_from_f1_list:.2f}",          f"±{err_from_f1_list:.2f}"],
        ["Specificity",  f"{mean_from_spec_list:.2f}", f"±{err_from_spec_list:.2f}"],

    ]

    # Print title
    print("\n" + table_title + "\n")

    # Print formatted table
    print("\n" + tabulate(
        table_data,
        headers=["Metric", "Score", "± Error"],
        tablefmt="fancy_grid",
        colalign=("left", "center", "center")
    ))

    return mean_from_acc_list, mean_from_prec_list, mean_from_rec_list, mean_from_f1_list, mean_from_spec_list, err_from_acc_list, err_from_prec_list, err_from_rec_list, err_from_f1_list, err_from_spec_list

def plot_performance_bar_chart(accuracy, precision, recall, f1, specificity, roc_auc, acc_err, prec_err, rec_err, f1_err, spec_err, auc_err):
    """

    Plot a bar chart with error bars for classification metrics.

    Parameters
    ----------
    accuracy : float
        Accuracy score.

    precision : float
        Precision score.

    recall : float
        Recall score.

    f1 : float
        F1-score.

    specificity : float
        Specificity score.

    roc_auc : float
        AUC score.

    acc_err : float
        Error margin for accuracy.

    prec_err : float
        Error margin for precision.

    rec_err : float
        Error margin for recall.

    f1_err : float
        Error margin for F1-score.

    spec_err : float
        Error margin for specificity.

    auc_err : float
        Error margin for AUC.

    Returns
    -------
    None

    """

    # Style settings for professional-IEEE plots
    matplotlib.rcParams.update({
        'font.size': 6,
        'font.family': 'serif',
        'axes.labelsize': 4,
        'axes.titlesize': 4,
        'legend.fontsize': 4,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'lines.linewidth': 0.8,
        'figure.dpi': 600,
        'savefig.dpi': 600
    })

    sns.set_palette("colorblind")
    logger.info("Generating performance metrics bar plot...")

    # Metrics and values
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUC']
    values = [accuracy, precision, recall, f1, specificity, roc_auc]
    errors = [acc_err, prec_err, rec_err, f1_err, spec_err, auc_err]
    colors = sns.color_palette("colorblind", n_colors=len(metrics))

    # Create figure
    fig, ax = plt.subplots(figsize=(3.2, 2.0))  # Wider for better label spacing

    # Plot bars with error bars
    bars = ax.bar(metrics, values, yerr=errors, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.7)

    # Aesthetics
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontweight='semibold')
    ax.set_title("Classification Metrics (± CI)", fontweight='bold', pad=5)
    ax.grid(axis='both', linestyle='--', linewidth=0.3, alpha=0.2)

    # Add value labels on top of bars
    #ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=4, fontweight='semibold')


    # Add value ± error labels above the error bars
    for bar, value, error in zip(bars, values, errors):
        top = value + error
        label = f"{value:.2f} ± {error:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + 0.05,  # piccolo margine sopra l'errore
            label,
            ha='center', va='bottom',
            fontsize=3, fontweight='semibold'
        )

    
    # Tick formatting
    ax.tick_params(axis='x', labelrotation=0)
    ax.tick_params(axis='both', direction='in', length=3, width=0.5)

    sns.despine(bottom=True)
    plt.tight_layout()
    plt.show()



