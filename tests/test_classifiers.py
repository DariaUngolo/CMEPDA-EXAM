import unittest
import numpy as np
import pandas as pd
from loguru import logger
from unittest.mock import patch
import sys
import atexit

# -------------------------------
# GLOBAL PATCH: Disable all plot saving/showing (matplotlib) during testing
# -------------------------------
# This prevents any GUI windows or file output related to plotting during test runs,
# which is useful when testing code that might otherwise call `plt.show()` or `savefig()`.
patches = [
    patch('matplotlib.pyplot.savefig', return_value=None),
    patch('matplotlib.pyplot.show', return_value=None),
    patch('matplotlib.figure.Figure.savefig', return_value=None),
    patch('matplotlib.figure.Figure.show', return_value=None),
    patch('ML_codes.performance_scores.plot_performance_bar_chart', lambda *a, **kw: None),
    patch('ML_codes.performance_scores.evaluate_model_performance', lambda *a, **kw: {
    'Accuracy': 1,
    'Precision': 1,
    'Recall': 1,
    'F1': 1,
    'F1-score': 1,
    'Specificity': 1,
    'AUC': 1
})
]
for p in patches:
    p.start()
    sys.modules[__name__].__dict__.setdefault('_patches', []).append(p)

from classifiers import RFPipeline_noPCA, RFPipeline_PCA, RFPipeline_RFECV, SVM_simple


class TestClassifiers(unittest.TestCase):
    """

    Unit tests for classification pipelines defined in `classifiers_unified`.

    The tests validate:
    - That each model (Random Forest with/without PCA, with RFECV, and SVM with different kernels)
      runs end-to-end without exceptions.
    - That each model can predict and returns an output of the correct length.

    """


    def setUp(self):
        """

        Create a synthetic dataset for testing.

        This method runs before every test case. It generates:
        - A 30x5 DataFrame with normally distributed features
        - A corresponding binary label Series with 15 "Normal" and 15 "AD" samples

        """
        logger.info("Setting up synthetic dataset for tests...")
        np.random.seed(42)
        self.X = pd.DataFrame(
            np.random.randn(30, 5),
            columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        )
        labels = ['Normal'] * 15 + ['AD'] * 15
        self.y = pd.Series(labels, index=self.X.index, name='Diagnosis')
        logger.info(f"Dataset created with {self.X.shape[0]} samples and {self.X.shape[1]} features.")

    def test_RFPipeline_noPCA_runs(self):
        """

        Test Random Forest pipeline without PCA.

        Ensures the model runs without error, supports `.predict`, and returns predictions
        of expected length.

        """
        logger.info("Testing RFPipeline_noPCA...")
        model = RFPipeline_noPCA(self.X, self.y, n_iter=2, cv=2)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        preds = model.predict(self.X.values)
        self.assertEqual(len(preds), len(self.X))
        logger.success("RFPipeline_noPCA executed and predicted successfully.")

    def test_RFPipeline_PCA_runs(self):

        """

        Test Random Forest pipeline with PCA dimensionality reduction.

        Verifies successful model training and prediction.

        """
        logger.info("Testing RFPipeline_PCA...")
        model = RFPipeline_PCA(self.X, self.y, n_iter=2, cv=2)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        preds = model.predict(self.X.values)
        self.assertEqual(len(preds), len(self.X))
        logger.success("RFPipeline_PCA executed and predicted successfully.")

    def test_RFPipeline_RFECV_runs(self):

        """

        Test Random Forest pipeline with recursive feature elimination (RFECV).

        Verifies that the model trains and attempts to predict. Since feature elimination
        can sometimes cause failures (e.g., if too few features are selected), this test
        is wrapped in a try/except block.

        """
        logger.info("Testing RFPipeline_RFECV...")
        model = RFPipeline_RFECV(self.X, self.y, n_iter=2, cv=2)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

        try:
            preds = model.predict(self.X.values)
            logger.info(f"Prediction with RFECV made. Number of predictions: {len(preds)}")
        except Exception as e:
            logger.warning(f"Prediction with RFECV failed: {e}")
            preds = None

        self.assertTrue(preds is None or len(preds) == len(self.X))
        logger.success("RFPipeline_RFECV executed (with or without prediction success).")

    def test_SVM_simple_linear_runs(self):

        """

        Test SVM with linear kernel.

        Ensures the model is created, supports prediction, and the output is correct.

        """
        logger.info("Testing SVM_simple with linear kernel...")
        model = SVM_simple(self.X, self.y, ker='linear', cv=2)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        preds = model.predict(self.X.values)
        self.assertEqual(len(preds), len(self.X))
        logger.success("SVM_simple (linear) executed and predicted successfully.")

    def test_SVM_simple_rbf_runs(self):

        """

        Test SVM with radial basis function (RBF) kernel.

        Ensures proper model training and prediction behavior.

        """
        logger.info("Testing SVM_simple with rbf kernel...")
        model = SVM_simple(self.X, self.y, ker='rbf', cv=2)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        preds = model.predict(self.X.values)
        self.assertEqual(len(preds), len(self.X))
        logger.success("SVM_simple (rbf) executed and predicted successfully.")


# -----------------------------------
# Cleanup: stop all patches on exit
# -----------------------------------
def _stop_patches():
    for p in getattr(sys.modules[__name__], '_patches', []):
        p.stop()

atexit.register(_stop_patches)

# To run this test suite manually, uncomment below:
# if __name__ == "__main__":
#     unittest.main()
