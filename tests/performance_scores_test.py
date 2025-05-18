import unittest
import numpy as np
from performance_scores import evaluate_model_performance

class TestEvaluatePerformance(unittest.TestCase):

    def test_perfect_classification(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        result = evaluate_model_performance(y_true, y_pred, y_proba)
        self.assertIn('accuracy', result)
        self.assertAlmostEqual(result['accuracy'][0], 1.0)
        self.assertAlmostEqual(result['precision'][0], 1.0)
        self.assertAlmostEqual(result['recall'][0], 1.0)
        self.assertAlmostEqual(result['f1_score'][0], 1.0)
        self.assertAlmostEqual(result['auc'][0], 1.0)
        # Specificity == 1 in this perfect case
        self.assertAlmostEqual(result['specificity'][0], 1.0)

    def test_total_misclassification(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.2])

        result = evaluate_model_performance(y_true, y_pred, y_proba)
        self.assertIn('accuracy', result)
        self.assertAlmostEqual(result['accuracy'][0], 0.0)
        # Precision, Recall, F1 score may be zero or undefined, test at least accuracy zero
        self.assertAlmostEqual(result['auc'][0], 0.0)

    def test_only_negative_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.3])

        result = evaluate_model_performance(y_true, y_pred, y_proba)
        self.assertIn('accuracy', result)
        # Accuracy should be computed normally
        self.assertGreaterEqual(result['accuracy'][0], 0.0)
        # Check keys exist
        for key in ['precision', 'recall', 'f1_score', 'specificity', 'auc']:
            self.assertIn(key, result)

    def test_only_positive_class(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_proba = np.array([0.9, 0.8, 0.1, 0.95])

        result = evaluate_model_performance(y_true, y_pred, y_proba)
        self.assertIn('accuracy', result)
        self.assertGreaterEqual(result['accuracy'][0], 0.0)
        for key in ['precision', 'recall', 'f1_score', 'specificity', 'auc']:
            self.assertIn(key, result)

    def test_single_sample_input(self):
        y_true = np.array([1])
        y_pred = np.array([1])
        y_proba = np.array([0.95])

        result = evaluate_model_performance(y_true, y_pred, y_proba)
        self.assertIn('accuracy', result)
        self.assertIsInstance(result['accuracy'][0], float)
        for key in ['precision', 'recall', 'f1_score', 'specificity', 'auc']:
            self.assertIn(key, result)

    def test_single_class_in_true_labels(self):
        # y_true all zeros, y_pred all zeros
        y_true = np.zeros(5, dtype=int)
        y_pred = np.zeros(5, dtype=int)
        y_proba = np.zeros(5, dtype=float)

        result = evaluate_model_performance(y_true, y_pred, y_proba)
        self.assertIn('accuracy', result)
        self.assertGreaterEqual(result['accuracy'][0], 0.0)

if __name__ == "__main__":
    unittest.main()


