import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from loguru import logger
import sys 
import os

# Ensure the 'ML_codes' directory is in the Python path for importing the feature_extractor module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML_codes')))
import feature_extractor  # Import the module (not just the function) to allow correct mocking


class TestFeatureExtractorMocked(unittest.TestCase):
    """
    Unit test class for the `feature_extractor()` function using the `unittest` framework,
    enhanced with `loguru` logging and mocking via `unittest.mock`.

    This class tests the behavior of the `feature_extractor` function by replacing it with a mock
    that returns synthetic data, avoiding the need for actual file I/O or MATLAB calls.

    Attributes:
        mock_df_mean (pd.DataFrame): Mocked dataframe of mean values for each subject and ROI.
        mock_df_std (pd.DataFrame): Mocked dataframe of standard deviations for each subject and ROI.
        mock_df_volume (pd.DataFrame): Mocked dataframe of volumes for each subject and ROI.
        mock_function (MagicMock): The mocked version of `feature_extractor.feature_extractor`.
        result (tuple): The tuple returned by the mocked function, containing mock DataFrames.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up mock return values and patch the feature_extractor function before running any tests.

        This method:
        - Creates synthetic DataFrames to simulate mean, std, and volume outputs.
        - Patches the actual `feature_extractor.feature_extractor()` function with a mock.
        - Calls the patched function to retrieve mock results.
        """
        logger.info("Setting up TestFeatureExtractorMocked...")

        num_subjects = 5
        num_rois = 10
        columns = [f'ROI_{i}' for i in range(num_rois)]

        cls.mock_df_mean = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)
        cls.mock_df_std = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)
        cls.mock_df_volume = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)

        logger.debug(f"Generated mock DataFrames with shape: {cls.mock_df_mean.shape}")

        # Patch the feature_extractor function to return mock data
        cls.patcher = patch('feature_extractor.feature_extractor', return_value=(
            cls.mock_df_mean,
            cls.mock_df_std,
            cls.mock_df_volume,
            None  # Represents an unused or null fourth return value
        ))

        cls.mock_function = cls.patcher.start()
        logger.info("feature_extractor() has been patched with mock return values.")

        # Call the patched function to capture the mocked results
        cls.result = feature_extractor.feature_extractor(
            folder_path='dummy_folder',
            atlas_file='dummy_atlas.nii.gz',
            atlas_txt='dummy_labels.txt',
            metadata_csv='dummy_metadata.tsv',
            output_csv_prefix='dummy_output',
            matlab_feature_extractor_path='dummy_matlab_path'
        )
        logger.info("Mocked feature_extractor() has been called.")

        # Unpack the returned dataframes
        cls.df_mean, cls.df_std, cls.df_volume, *_ = cls.result

    @classmethod
    def tearDownClass(cls):
        """
        Tear down the mock by stopping the patcher after all tests have run.
        """
        cls.patcher.stop()
        logger.info("TearDown complete. feature_extractor() patch removed.")

    def test_shape_consistency(self):
        """
        Test to ensure the shapes of all returned DataFrames are consistent.

        Verifies that the number of subjects and ROIs is equal across mean, std, and volume outputs.
        """
        logger.info("Running test_shape_consistency...")
        self.assertEqual(self.df_mean.shape, self.df_std.shape)
        self.assertEqual(self.df_mean.shape, self.df_volume.shape)
        logger.success("test_shape_consistency passed.")

    def test_not_empty(self):
        """
        Test to ensure that the returned DataFrames are not empty.

        Checks that the DataFrames contain at least one row and one column.
        """
        logger.info("Running test_not_empty...")
        self.assertGreater(self.df_mean.shape[0], 0)
        self.assertGreater(self.df_mean.shape[1], 0)
        logger.success("test_not_empty passed.")

    def test_column_names(self):
        """
        Test to verify that all DataFrames have matching column names.

        Ensures that ROI names are consistent across mean, std, and volume DataFrames.
        """
        logger.info("Running test_column_names...")
        self.assertListEqual(list(self.df_mean.columns), list(self.df_std.columns))
        self.assertListEqual(list(self.df_std.columns), list(self.df_volume.columns))
        logger.success("test_column_names passed.")

    def test_mock_was_called(self):
        """
        Test to confirm that the mocked function was called exactly once.

        Validates that the patching mechanism correctly intercepted the function call.
        """
        logger.info("Running test_mock_was_called...")
        self.mock_function.assert_called_once()
        logger.success("test_mock_was_called passed.")


if __name__ == '__main__':
    logger.info("Starting unittest for feature extractor (mocked path for Matlab)...")
    unittest.main()
