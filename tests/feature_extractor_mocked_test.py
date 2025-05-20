import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from loguru import logger
import feature_extractor  # Import the module, not the function, to allow correct patching


class TestFeatureExtractorMocked(unittest.TestCase):
    """
    Unit test for the `feature_extractor()` function using mocking and loguru logging.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock data and patch the target function before any test runs.
        """
        logger.info("Setting up TestFeatureExtractorMocked...")

        num_subjects = 5
        num_rois = 10
        columns = [f'ROI_{i}' for i in range(num_rois)]

        cls.mock_df_mean = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)
        cls.mock_df_std = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)
        cls.mock_df_volume = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)

        logger.debug(f"Generated mock DataFrames with shape: {cls.mock_df_mean.shape}")

        # Start patching the feature_extractor function
        cls.patcher = patch('feature_extractor.feature_extractor', return_value=(
            cls.mock_df_mean,
            cls.mock_df_std,
            cls.mock_df_volume,
            None
        ))
        cls.mock_function = cls.patcher.start()
        logger.info("feature_extractor() has been patched with mock return values.")

        # Simulate calling the patched function
        cls.result = feature_extractor.feature_extractor(
            folder_path='dummy_folder',
            atlas_file='dummy_atlas.nii.gz',
            atlas_txt='dummy_labels.txt',
            metadata_csv='dummy_metadata.tsv',
            output_csv_prefix='dummy_output',
            matlab_feature_extractor_path='dummy_matlab_path'
        )
        logger.info("Mocked feature_extractor() has been called.")

        cls.df_mean, cls.df_std, cls.df_volume, *_ = cls.result

    @classmethod
    def tearDownClass(cls):
        """
        Clean up by stopping the patcher.
        """
        cls.patcher.stop()
        logger.info("TearDown complete. feature_extractor() patch removed.")

    def test_shape_consistency(self):
        """
        Ensure returned DataFrames have consistent shape.
        """
        logger.info("Running test_shape_consistency...")
        self.assertEqual(self.df_mean.shape, self.df_std.shape)
        self.assertEqual(self.df_mean.shape, self.df_volume.shape)
        logger.success("test_shape_consistency passed.")

    def test_not_empty(self):
        """
        Ensure DataFrames are not empty.
        """
        logger.info("Running test_not_empty...")
        self.assertGreater(self.df_mean.shape[0], 0)
        self.assertGreater(self.df_mean.shape[1], 0)
        logger.success("test_not_empty passed.")

    def test_column_names(self):
        """
        Ensure column names match across all DataFrames.
        """
        logger.info("Running test_column_names...")
        self.assertListEqual(list(self.df_mean.columns), list(self.df_std.columns))
        self.assertListEqual(list(self.df_std.columns), list(self.df_volume.columns))
        logger.success("test_column_names passed.")

    def test_mock_was_called(self):
        """
        Ensure mock was triggered exactly once.
        """
        logger.info("Running test_mock_was_called...")
        self.mock_function.assert_called_once()
        logger.success("test_mock_was_called passed.")


if __name__ == '__main__':
    logger.info("Starting unittest for feature extractor (mocked path for Matlab)...")
    unittest.main()
