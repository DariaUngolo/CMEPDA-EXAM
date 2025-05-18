import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from performance_scores import compute_binomial_error, evaluate_model_performance
from loguru import logger
from unittest.mock import patch
import sys
import atexit

# Optional: Log to file with rotation and rich diagnostics
logger.add("test_logs.log", level="DEBUG", rotation="500 KB", backtrace=True, diagnose=True)

# Patching matplotlib to suppress display/save operations during test execution
patches = [
    patch('matplotlib.pyplot.savefig', return_value=None),
    patch('matplotlib.pyplot.show', return_value=None),
    patch('matplotlib.figure.Figure.savefig', return_value=None),
    patch('matplotlib.figure.Figure.show', return_value=None),
]
for p in patches:
    p.start()
    sys.modules[__name__].__dict__.setdefault('_patches', []).append(p)

class TestComputeBinomialError(unittest.TestCase):
    """
    Unit tests for the compute_binomial_error function.
    These tests cover the core statistical behavior of the confidence interval estimation.
    """

    def test_error_mid_value(self):
        """
        Test the error for p=0.5 and n=100.
        This represents the maximum uncertainty scenario.
        """
        logger.info("Running test: test_error_mid_value")
        err = compute_binomial_error(0.5, 100, 0.683)
        logger.debug(f"Computed error: {err}")
        self.assertAlmostEqual(err, 0.05, places=2)
        logger.success("test_error_mid_value passed.")

    def test_error_zero_probability(self):
        """
        Test the error for a perfect classifier (p=0.0).
        The error bound should be zero.
        """
        logger.info("Running test: test_error_zero_probability")
        err = compute_binomial_error(0.0, 100, 0.95)
        logger.debug(f"Computed error: {err}")
        self.assertEqual(err, 0.0)
        logger.success("test_error_zero_probability passed.")

    def test_error_full_probability(self):
        """
        Test the error for p=1.0.
        Again, the error should be zero for a perfect classifier.
        """
        logger.info("Running test: test_error_full_probability")
        err = compute_binomial_error(1.0, 100, 0.95)
        logger.debug(f"Computed error: {err}")
        self.assertEqual(err, 0.0)
        logger.success("test_error_full_probability passed.")

    def test_error_invalid_n_samples(self):
        """
        Test that passing n=0 (invalid number of samples) raises a ZeroDivisionError.
        """
        logger.info("Running test: test_error_invalid_n_samples")
        with self.assertRaises(ZeroDivisionError):
            compute_binomial_error(0.5, 0, 0.95)
        logger.success("test_error_invalid_n_samples passed.")


class TestEvaluateModelPerformance(unittest.TestCase):
    """
    Unit tests for the evaluate_model_performance function.
    These tests validate the correctness of metrics, confidence intervals,
    and input flexibility (e.g., accepting both 1D and 2D probability arrays).
    """

    def setUp(self):
        """
        Create dummy prediction labels and probabilities for classification performance testing.
        """
        logger.info("Setting up test data for TestEvaluateModelPerformance")
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        self.y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.3, 0.4, 0.1, 0.95, 0.88])

    def test_output_keys(self):
        """
        Ensure all expected metric and error keys are present in the output.
        """
        logger.info("Running test: test_output_keys")
        results = evaluate_model_performance(self.y_true, self.y_pred, self.y_proba, confidence_level=0.683)
        logger.debug(f"Returned keys: {list(results.keys())}")
        expected_keys = [
            'Accuracy', 'Accuracy_error',
            'Precision', 'Precision_error',
            'Recall', 'Recall_error',
            'F1-score', 'F1-score_error',
            'Specificity', 'Specificity_error',
            'AUC', 'AUC_error'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        logger.success("test_output_keys passed.")

    def test_metric_values_range(self):
        """
        Check that all metric values (excluding errors) fall between 0.0 and 1.0.
        """
        logger.info("Running test: test_metric_values_range")
        results = evaluate_model_performance(self.y_true, self.y_pred, self.y_proba, confidence_level=0.683)
        for key, value in results.items():
            logger.debug(f"{key}: {value}")
            if 'error' not in key:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        logger.success("test_metric_values_range passed.")

    def test_auc_close_to_sklearn(self):
        """
        Ensure the computed AUC matches scikit-learn's AUC within acceptable tolerance.
        """
        logger.info("Running test: test_auc_close_to_sklearn")
        results = evaluate_model_performance(self.y_true, self.y_pred, self.y_proba)
        sklearn_auc = roc_auc_score(self.y_true, self.y_proba)
        logger.debug(f"Model AUC: {results['AUC']}, sklearn AUC: {sklearn_auc}")
        self.assertAlmostEqual(results['AUC'], sklearn_auc, places=2)
        logger.success("test_auc_close_to_sklearn passed.")

    def test_y_proba_2d_input(self):
        """
        Ensure that the function correctly handles 2D probability arrays (e.g., from softmax).
        """
        logger.info("Running test: test_y_proba_2d_input")
        y_proba_2d = np.column_stack([1 - self.y_proba, self.y_proba])
        results = evaluate_model_performance(self.y_true, self.y_pred, y_proba_2d)
        logger.debug(f"AUC from 2D proba: {results['AUC']}")
        self.assertIn('AUC', results)
        logger.success("test_y_proba_2d_input passed.")


# Clean-up: stop all matplotlib patches when the script exits
def _stop_patches():
    for p in getattr(sys.modules[__name__], '_patches', []):
        p.stop()
atexit.register(_stop_patches)

if __name__ == '__main__':
    logger.info("Starting performance scores test...")
    unittest.main()
