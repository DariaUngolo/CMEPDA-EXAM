import unittest
import pandas as pd
import numpy as np

from classifiers_unified import RFPipeline_noPCA, RFPipeline_PCA, RFPipeline_RFECV, SVM_simple


class TestMLPipelines(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Creo un dataset di esempio piccolo, 20 campioni, 5 feature
        np.random.seed(42)
        cls.features = pd.DataFrame(
            np.random.randn(20, 5),
            columns=[f"ROI_{i}" for i in range(5)]
        )
        # Etichette casuali 'Normal' e 'AD'
        labels = np.random.choice(['Normal', 'AD'], size=20)
        cls.labels = pd.DataFrame(labels, index=cls.features.index, columns=['Diagnosis'])
    
    def test_RFPipeline_noPCA_runs(self):
        pipeline = RFPipeline_noPCA(self.features, self.labels, n_iter=5, cv=3)
        # Verifico che il ritorno sia una Pipeline sklearn
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "fit"))
        
    def test_RFPipeline_PCA_runs(self):
        pipeline = RFPipeline_PCA(self.features, self.labels, n_iter=5, cv=3)
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "fit"))
        
    def test_RFPipeline_RFECV_runs(self):
        pipeline = RFPipeline_RFECV(self.features, self.labels, n_iter=5, cv=3)
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "fit"))
        
    def test_SVM_simple_runs_linear(self):
        grid = SVM_simple(self.features, self.labels, ker='linear')
        self.assertTrue(hasattr(grid, "predict"))
        self.assertTrue(hasattr(grid, "fit"))
        
    def test_SVM_simple_runs_rbf(self):
        grid = SVM_simple(self.features, self.labels, ker='rbf')
        self.assertTrue(hasattr(grid, "predict"))
        self.assertTrue(hasattr(grid, "fit"))


if __name__ == '__main__':
    unittest.main()
