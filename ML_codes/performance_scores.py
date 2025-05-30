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
    logger.debug(f"Computing binomial error: z={z:.4f}, metric_value={metric_value:.4f}, n_samples={n_samples}")
    return z * np.sqrt((metric_value * (1 - metric_value)) / n_samples)


def evaluate_model_performance(y_true, y_pred, y_proba, confidence_level=0.683):
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

    y_proba : array-like
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
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
        logger.debug("Converted 2D array of probabilities to 1D (positive class).")

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division='warn')
    recall = recall_score(y_true, y_pred, zero_division='warn')
    f1 = f1_score(y_true, y_pred, zero_division='warn')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.debug(f"Confusion matrix values: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logger.debug("Computing error margins for classification metrics...")

    # Estimate errors using normal approximation
    n = len(y_true)
    acc_err = compute_binomial_error(accuracy, n, confidence_level)
    prec_err = compute_binomial_error(precision, n, confidence_level)
    rec_err = compute_binomial_error(recall, n, confidence_level)
    f1_err = compute_binomial_error(f1, n, confidence_level)
    spec_err = compute_binomial_error(specificity, n, confidence_level)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    logger.debug(f"ROC AUC: {roc_auc:.4f}")

    # Estimate AUC error using Hanley & McNeil method
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc**2 / (1 + roc_auc)
    z = norm.ppf((1 + confidence_level) / 2.0)
    auc_err = z * np.sqrt(
        (roc_auc * (1 - roc_auc) +
         (n1 - 1) * (q1 - roc_auc ** 2) +
         (n0 - 1) * (q2 - roc_auc ** 2)) / (n1 * n0)
    )
    logger.debug(f"AUC error estimated: ±{auc_err:.4f}")

    # === PLOTS ===

    # Configure matplotlib for IEEE-style plots
    matplotlib.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'axes.labelsize': 6,
        'axes.titlesize': 7,
        'legend.fontsize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.3,
        'figure.dpi': 600
    })
    sns.set_palette("colorblind")

    # Plot ROC curve
    logger.info("Generating ROC curve plot...")
    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f} ± {auc_err:.2f}', color='tab:blue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=0.8)
    ax.set_xlabel('False Positive Rate', fontsize=4, fontweight='semibold', labelpad=2)
    ax.set_ylabel('True Positive Rate', fontsize=4, fontweight='semibold')
    ax.set_title('ROC Curve', pad=4, fontsize=5, fontweight='bold')
    ax.legend(loc='lower right', frameon=False, fontsize=5)
    ax.tick_params(axis='both', labelsize=3.8)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Plot bar chart with error bars for each metric
    logger.info("Generating metrics bar plot...")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUC']
    values = [accuracy, precision, recall, f1, specificity, roc_auc]
    errors = [acc_err, prec_err, rec_err, f1_err, spec_err, auc_err]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    bars = ax.bar(metrics, values, yerr=errors, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.6)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=6, fontweight='semibold')
    ax.set_title("Classification Metrics with Confidence Intervals", fontsize=4, fontweight='bold', pad=6)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=4, fontweight='semibold')
    plt.xticks(fontsize=4, fontweight='semibold', rotation=12, ha='right')
    plt.yticks(fontsize=5, fontweight='semibold')
    sns.despine(bottom=True)
    plt.subplots_adjust(bottom=0.18, left=0.10, right=0.98, top=0.90)
    plt.tight_layout()
    plt.show()

    from tabulate import tabulate

    # Prepare table data
    table_data = [
        ["Accuracy",     f"{accuracy:.2f}",    f"±{acc_err:.2f}"],
        ["Precision",    f"{precision:.2f}",   f"±{prec_err:.2f}"],
        ["Recall",       f"{recall:.2f}",      f"±{rec_err:.2f}"],
        ["F1-score",     f"{f1:.2f}",          f"±{f1_err:.2f}"],
        ["Specificity",  f"{specificity:.2f}", f"±{spec_err:.2f}"],
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
        'Accuracy': accuracy, 'Accuracy_error': acc_err,
        'Precision': precision, 'Precision_error': prec_err,
        'Recall': recall, 'Recall_error': rec_err,
        'F1-score': f1, 'F1-score_error': f1_err,
        'Specificity': specificity, 'Specificity_error': spec_err,
        'AUC': roc_auc, 'AUC_error': auc_err
    }
