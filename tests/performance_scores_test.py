import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from performance_scores import compute_binomial_error, evaluate_model_performance
from loguru import logger

# Patch matplotlib per bloccare la visualizzazione/salvataggio dei grafici durante i test
from unittest.mock import patch
import sys

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

    def test_error_mid_value(self):
        logger.info("Eseguendo test_error_mid_value")
        err = compute_binomial_error(0.5, 100, 0.683)
        logger.debug(f"Errore calcolato: {err}")
        self.assertAlmostEqual(err, 0.05, places=2)

    def test_error_zero_probability(self):
        logger.info("Eseguendo test_error_zero_probability")
        err = compute_binomial_error(0.0, 100, 0.95)
        logger.debug(f"Errore calcolato: {err}")
        self.assertEqual(err, 0.0)

    def test_error_full_probability(self):
        logger.info("Eseguendo test_error_full_probability")
        err = compute_binomial_error(1.0, 100, 0.95)
        logger.debug(f"Errore calcolato: {err}")
        self.assertEqual(err, 0.0)

    def test_error_invalid_n_samples(self):
        logger.info("Eseguendo test_error_invalid_n_samples (valore non valido di n)")
        with self.assertRaises(ZeroDivisionError):
            compute_binomial_error(0.5, 0, 0.95)


class TestEvaluateModelPerformance(unittest.TestCase):

    def setUp(self):
        logger.info("Setup dei dati di test")
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        self.y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.3, 0.4, 0.1, 0.95, 0.88])

    def test_output_keys(self):
        logger.info("Eseguendo test_output_keys")
        results = evaluate_model_performance(self.y_true, self.y_pred, self.y_proba, confidence_level=0.683)
        logger.debug(f"Chiavi ottenute: {list(results.keys())}")
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

    def test_metric_values_range(self):
        logger.info("Eseguendo test_metric_values_range")
        results = evaluate_model_performance(self.y_true, self.y_pred, self.y_proba, confidence_level=0.683)
        for key, value in results.items():
            logger.debug(f"{key}: {value}")
            if 'error' not in key:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_auc_close_to_sklearn(self):
        logger.info("Eseguendo test_auc_close_to_sklearn")
        results = evaluate_model_performance(self.y_true, self.y_pred, self.y_proba)
        sklearn_auc = roc_auc_score(self.y_true, self.y_proba)
        logger.debug(f"AUC modello: {results['AUC']}, AUC sklearn: {sklearn_auc}")
        self.assertAlmostEqual(results['AUC'], sklearn_auc, places=2)

    def test_y_proba_2d_input(self):
        logger.info("Eseguendo test_y_proba_2d_input")
        y_proba_2d = np.column_stack([1 - self.y_proba, self.y_proba])
        results = evaluate_model_performance(self.y_true, self.y_pred, y_proba_2d)
        logger.debug(f"AUC con proba 2D: {results['AUC']}")
        self.assertIn('AUC', results)

import atexit
def _stop_patches():
    for p in getattr(sys.modules[__name__], '_patches', []):
        p.stop()
atexit.register(_stop_patches)

if __name__ == '__main__':
    logger.info("Avvio dei test...")
    unittest.main()