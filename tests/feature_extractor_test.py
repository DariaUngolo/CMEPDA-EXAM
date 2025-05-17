import unittest
import argparse
import os
import sys
import numpy as np

# Add the project root folder to PYTHONPATH to import feature_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor import feature_extractor


class TestFeatureExtractorMATLAB(unittest.TestCase):
    """
    Unit test suite for the MATLAB-based feature_extractor function.

    This test class:
    - Checks that the required MATLAB script exists.
    - Runs the feature extraction once before all tests (setUpClass).
    - Verifies that the output dataframes are valid, consistent, and contain finite numeric values.

    Input arguments are parsed from command line (before unittest execution) to
    provide the necessary paths and parameters to the feature extractor.
    """

    @classmethod
    def setUpClass(cls):
        """
        Class-level setup, runs once before all tests.

        - Checks if the MATLAB .m file exists in the specified path.
        - Executes the feature_extractor function, storing the output dataframes as class variables.
        """
        # Check if the MATLAB function file exists
        expected_file = os.path.join(cls.args.matlab_path, 'feature_extractor.m')
        if not os.path.exists(expected_file):
            raise FileNotFoundError(f"❌ MATLAB file '{expected_file}' not found.")

        # Run feature extraction once for all tests
        result = feature_extractor(
            folder_path=cls.args.folder_path,
            atlas_file=cls.args.atlas_file,
            atlas_txt=cls.args.atlas_txt,
            metadata_csv=cls.args.metadata_csv,
            output_csv_prefix=cls.args.output_prefix,
            matlab_feature_extractor_path=cls.args.matlab_path
        )

        # Unpack results into class variables for use in tests
        cls.df_mean, cls.df_std, cls.df_volume, *_ = result

    def test_dataframe_not_empty(self):
        """Check that the mean features DataFrame is not empty."""
        self.assertGreater(self.df_mean.shape[0], 0, "❌ The DataFrame `df_mean` is empty.")

    def test_shapes_are_equal(self):
        """Check that mean and std DataFrames have the same shape."""
        self.assertEqual(
            self.df_mean.shape, self.df_std.shape,
            "❌ Shapes of `df_mean` and `df_std` do not match."
        )

    def test_volume_matches_subjects(self):
        """Check that the number of rows in df_volume matches the number of subjects (rows in df_mean)."""
        self.assertEqual(
            self.df_volume.shape[0], self.df_mean.shape[0],
            "❌ Number of subjects in `df_volume` does not match `df_mean`."
        )

    def test_values_are_finite(self):
        """Verify all values in the DataFrames are finite (no NaN or Inf)."""
        for df, name in zip([self.df_mean, self.df_std, self.df_volume], ["mean", "std", "volume"]):
            self.assertTrue(
                np.isfinite(df.values).all(),
                f"❌ The DataFrame `{name}` contains NaN or infinite values."
            )


if __name__ == '__main__':
    # Parse arguments BEFORE unittest.main()
    parser = argparse.ArgumentParser(description="Unit tests for MATLAB feature_extractor function.")
    parser.add_argument('--folder_path', required=True, help="Folder with subject NIfTI images")
    parser.add_argument('--atlas_file', required=True, help="Resampled atlas NIfTI file")
    parser.add_argument('--atlas_txt', required=True, help="TXT file with ROI labels")
    parser.add_argument('--metadata_csv', required=True, help="TSV metadata file with subject IDs and diagnosis")
    parser.add_argument('--output_prefix', required=True, help="Prefix for MATLAB output CSV files")
    parser.add_argument('--matlab_path', required=True, help="Path to folder containing feature_extractor.m")

    args, remaining_args = parser.parse_known_args()

    # Inject parsed args into the test class
    TestFeatureExtractorMATLAB.args = args

    # Run unittest with any leftover args (allowing unittest options)
    unittest.main(argv=[sys.argv[0]] + remaining_args)


