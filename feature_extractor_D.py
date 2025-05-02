import numpy as np
import pandas as pd
import matlab.engine
from pathlib import Path
import os
import sys

def feature_extractor(image_filepaths, atlas_file, atlas_txt):
    """
    Extracts region-wise features (mean and standard deviation of voxel intensities)
    from a list of NIfTI brain images using a MATLAB function that leverages an atlas.

    This function communicates with MATLAB through the MATLAB Engine API for Python
    and executes a MATLAB function called 'feature_extractor.m', which performs the
    actual feature extraction using a labeled brain atlas.

    Parameters
    ----------
    image_filepaths : list of str
        Full paths to the NIfTI image files (e.g., smwc1*.nii).
    atlas_file : str
        Full path to the brain atlas file in NIfTI format (.nii or .nii.gz).
        This file defines the anatomical brain regions.
    atlas_txt : str
        Path to a text file listing the ROI IDs and corresponding region names.
        Expected format: one line per region, with tab-separated values (ID<TAB>RegionName).

    Returns
    -------
    df_mean : pandas.DataFrame
        DataFrame where rows correspond to subjects (images) and columns to brain regions.
        Each value represents the mean voxel intensity within a given ROI for a subject.
    df_std : pandas.DataFrame
        DataFrame with the same structure as df_mean, containing standard deviations.
    group : pandas.Series
        Series containing the diagnostic label (e.g., AD or control) for each subject.
        The index of this series matches the index of the returned DataFrames.

    Notes
    -----
    - The function requires MATLAB to be installed and accessible from Python.
    - The MATLAB function 'feature_extractor.m' must be in a folder added to the MATLAB path.
    - An external CSV file named 'AD_CTRL_metadata.csv' is required to map subjects to labels.
      It must contain an 'ID' column (matching image IDs) and a 'DXGROUP' column for labels.
    """

    # Start a new MATLAB session
    eng = matlab.engine.start_matlab()

    # Add the folder containing the MATLAB function to the MATLAB path
    eng.addpath(r'C:\Users\brand\OneDrive\Desktop\CMEPDA EXAM\CMEPDA-EXAM')

    # Convert image paths to MATLAB-compatible string array
    image_paths_matlab = matlab.string(image_filepaths)

    # Call the MATLAB function and receive two outputs (mean and std tables)
    mean_mat, std_mat = eng.feature_extractor(image_paths_matlab, atlas_file, atlas_txt, nargout=2)

    # Stop MATLAB engine to free resources
    eng.quit()

    # Convert MATLAB outputs to NumPy arrays
    mean_np = np.array(mean_mat)
    std_np = np.array(std_mat)

    # Extract ROI names (column headers) and subject IDs (row indices)
    roi_names = mean_np[0, 1:]         # First row, from second column onward
    subject_ids = mean_np[1:, 0]       # First column, from second row onward

    # Extract numeric data from cell arrays (as strings) and convert to float
    data_mean = mean_np[1:, 1:].astype(float)
    data_std = std_np[1:, 1:].astype(float)

    # Create pandas DataFrames
    df_mean = pd.DataFrame(data_mean, columns=roi_names, index=subject_ids)
    df_std = pd.DataFrame(data_std, columns=roi_names, index=subject_ids)

    # Load metadata to extract diagnostic labels
    df_group = pd.read_csv("C:\\Users\\brand\\OneDrive\\Desktop\\CMEPDA\\progetto esame\\data\\AD_CTRL_metadata (1).csv")  # Must contain columns: 'ID', 'DXGROUP'
    df_group.set_index('ID', inplace=True)

    # Align subject labels with extracted features using subject IDs
    group = df_group.loc[df_mean.index, 'DXGROUP']

    return df_mean, df_std, group


