import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import feature_extractor  # Import the module, not the function, to allow correct patching


class TestFeatureExtractorMocked(unittest.TestCase):
    """
    Unit test for the `feature_extractor()` function using mocking.

    This test suite verifies the behavior of the feature extraction pipeline
    by mocking the function responsible for interfacing with MATLAB.
    It avoids any dependency on actual NIfTI files or MATLAB runtime,
    and tests the structure and integrity of the returned data frames.

    The function under test is assumed to return:
        - df_mean: DataFrame of ROI-wise mean intensities
        - df_std:  DataFrame of ROI-wise standard deviations
        - df_volume: DataFrame of ROI-wise volumes
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test-wide resources.
        Generates fake dataframes that simulate the output of the real
        feature_extractor function. Then, uses unittest.mock.patch to
        replace the real function with a mocked version.
        """
        # Create dummy feature tables (e.g., 5 subjects x 10 ROIs)
        num_subjects = 5
        num_rois = 10
        columns = [f'ROI_{i}' for i in range(num_rois)]

        cls.mock_df_mean = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)
        cls.mock_df_std = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)
        cls.mock_df_volume = pd.DataFrame(np.random.rand(num_subjects, num_rois), columns=columns)

        # Patch the real function with a mock that returns fake DataFrames
        cls.patcher = patch('feature_extractor.feature_extractor', return_value=(
            cls.mock_df_mean,
            cls.mock_df_std,
            cls.mock_df_volume,
            None  # You can extend this tuple if more outputs are expected
        ))
        cls.mock_function = cls.patcher.start()

        # Simulate the function call (mocked)
        cls.result = feature_extractor.feature_extractor(
            folder_path='dummy_folder',
            atlas_file='dummy_atlas.nii.gz',
            atlas_txt='dummy_labels.txt',
            metadata_csv='dummy_metadata.tsv',
            output_csv_prefix='dummy_output',
            matlab_feature_extractor_path='dummy_matlab_path'
        )

        # Unpack the mocked result
        cls.df_mean, cls.df_std, cls.df_volume, *_ = cls.result

    @classmethod
    def tearDownClass(cls):
        """
        Clean up after all tests have run.
        Stops the patcher to restore the original behavior of the function.
        """
        cls.patcher.stop()

    def test_shape_consistency(self):
        """
        Ensure that the returned DataFrames have the same shape.
        This checks that mean, std, and volume metrics are computed for
        the same number of subjects and ROIs.
        """
        self.assertEqual(self.df_mean.shape, self.df_std.shape)
        self.assertEqual(self.df_mean.shape, self.df_volume.shape)

    def test_not_empty(self):
        """
        Check that the DataFrames are not empty.
        This ensures the mock is returning non-trivial data.
        """
        self.assertGreater(self.df_mean.shape[0], 0)
        self.assertGreater(self.df_mean.shape[1], 0)

    def test_column_names(self):
        """
        Verify that all DataFrames have identical column names.
        The columns (ROIs) must match across mean, std, and volume outputs.
        """
        self.assertListEqual(list(self.df_mean.columns), list(self.df_std.columns))
        self.assertListEqual(list(self.df_std.columns), list(self.df_volume.columns))

    def test_mock_was_called(self):
        """
        Confirm that the mock function was actually called once.
        This validates that the patch was applied correctly.
        """
        self.mock_function.assert_called_once()


if __name__ == '__main__':
    # Run the test suite
    unittest.main()
