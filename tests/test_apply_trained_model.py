import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import pandas as pd
import sys
import os
from matlab import double as matlab_double
from loguru import logger

# Import your module here, e.g.:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML_codes')))
import apply_trained_model


class TestFeatureModule(unittest.TestCase):

    @patch("apply_trained_model.matlab.engine.start_matlab")
    @patch("builtins.open", new_callable=mock_open, read_data="1 ROI1\n2 ROI2\n3 ROI3\n")
    def test_feature_extractor_independent_dataset(self, mock_file, mock_start_matlab):

        logger.info("Starting test for feature_extractor_independent_dataset")

        # Mock MATLAB engine and feature_extractor output
        mock_eng = MagicMock()

        # Mock data arrays must have shape (3 rows, 3 columns) matching the 3 ROIs in atlas_txt


        mean = matlab_double([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        std = matlab_double([[0.1, 0.2, 0.3],
                     [0.4, 0.5, 0.6],
                     [0.7, 0.8, 0.9]])

        volume = matlab_double([[10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]])


        mock_eng.feature_extractor.return_value = (mean, std, volume)
        mock_start_matlab.return_value = mock_eng

        # Call the function under test
        df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume = \
            apply_trained_model.feature_extractor_independent_dataset(
                "fake_nifti_path.nii",
                "fake_atlas_file.nii",
                "fake_atlas_txt.txt",
                "fake_matlab_path"
            )

        # Check if dataframes have correct shapes and columns
        self.assertEqual(df_mean.shape, (3, 3))  # 3 rows, 3 columns for mean
        self.assertListEqual(list(df_mean.columns), ['ROI1', 'ROI2', 'ROI3'])

        self.assertEqual(df_std.shape, (3, 3))
        self.assertListEqual(list(df_std.columns), ['ROI1', 'ROI2', 'ROI3'])

        self.assertEqual(df_volume.shape, (3, 3))
        self.assertListEqual(list(df_volume.columns), ['ROI1', 'ROI2', 'ROI3'])

        # Also test combined dataframes columns
        self.assertListEqual(list(df_mean_std.columns), 
                             ['ROI1_mean', 'ROI2_mean', 'ROI3_mean',
                              'ROI1_std', 'ROI2_std', 'ROI3_std'])
        self.assertListEqual(list(df_mean_volume.columns), 
                             ['ROI1_mean', 'ROI2_mean', 'ROI3_mean',
                              'ROI1_volume', 'ROI2_volume', 'ROI3_volume'])
        self.assertListEqual(list(df_std_volume.columns),
                             ['ROI1_std', 'ROI2_std', 'ROI3_std',
                              'ROI1_volume', 'ROI2_volume', 'ROI3_volume'])
        self.assertListEqual(list(df_mean_std_volume.columns),
                             ['ROI1_mean', 'ROI2_mean', 'ROI3_mean',
                              'ROI1_std', 'ROI2_std', 'ROI3_std',
                              'ROI1_volume', 'ROI2_volume', 'ROI3_volume'])

        logger.info("Test for feature_extractor_independent_dataset passed successfully")


if __name__ == "__main__":
    unittest.main()

