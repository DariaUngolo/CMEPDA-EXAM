import argparse
import numpy as np
import pandas as pd
import joblib  # For loading the saved model
import matlab.engine
from pathlib import Path
import os
from loguru import logger


def feature_extractor_independent_dataset(nifti_image_path, atlas_file, atlas_txt, matlab_feature_extractor_path):
    """
    Extracts statistical features from a NIfTI image using a MATLAB-based pipeline.

    This function calls a MATLAB feature extraction function to compute mean intensity,
    standard deviation, and volume metrics for regions of interest (ROIs) defined by an atlas.
    The extracted features are returned as multiple pandas DataFrames combining these statistics.

    Parameters
    ----------
    nifti_image_path : str
        Path to the input NIfTI image file.
    atlas_file : str
        Path to the brain atlas NIfTI file defining ROIs.
    atlas_txt : str
        Path to the atlas labels text file listing ROI names.
    matlab_feature_extractor_path : str
        Path to the MATLAB directory or script containing the feature extraction function.

    Returns
    -------
    df_mean : pandas.DataFrame
        DataFrame with mean intensity values per ROI.
    df_std : pandas.DataFrame
        DataFrame with standard deviation of intensities per ROI.
    df_volume : pandas.DataFrame
        DataFrame with volume measurements per ROI.
    df_mean_std : pandas.DataFrame
        DataFrame combining mean and standard deviation features.
    df_mean_volume : pandas.DataFrame
        DataFrame combining mean and volume features.
    df_std_volume : pandas.DataFrame
        DataFrame combining standard deviation and volume features.
    df_mean_std_volume : pandas.DataFrame
        DataFrame combining mean, standard deviation, and volume features.

    Notes
    -----
    - The input NIfTI image and atlas must be spatially aligned.
    - The MATLAB engine for Python must be installed and configured.
    - The MATLAB feature extraction function is expected to return matrices of features.
    - ROI labels are extracted from the provided atlas text file to name DataFrame columns.
    """
    logger.info("Starting MATLAB engine.")
    eng = matlab.engine.start_matlab()

    logger.info("Adding MATLAB feature extractor path to MATLAB engine.")
    eng.addpath(matlab_feature_extractor_path, nargout=0)

    logger.info("Calling MATLAB function for feature extraction.")
    mean, std, volume = eng.feature_extractor(nifti_image_path, atlas_file, atlas_txt, nargout=3)

    logger.info("Stopping MATLAB engine.")
    eng.quit()

    logger.info("Converting MATLAB output to NumPy arrays.")
    mean_matrix = np.asarray(mean)
    std_matrix = np.asarray(std)
    volume_matrix = np.asarray(volume)

    logger.debug(f"Mean matrix shape: {mean_matrix.shape}")
    logger.debug(f"STD matrix shape: {std_matrix.shape}")
    logger.debug(f"Volume matrix shape: {volume_matrix.shape}")

    logger.info("Determining if the first row contains headers.")
    data_start = 0
    if isinstance(mean_matrix[0, 0], str):
        data_start = 1
    if isinstance(std_matrix[0, 0], str):
        data_start = 1
    if isinstance(volume_matrix[0, 0], str):
        data_start = 1

    logger.info("Concatenating feature matrices.")
    mean_std_matrix = np.hstack((mean_matrix, std_matrix))
    mean_volume_matrix = np.hstack((mean_matrix, volume_matrix))
    std_volume_matrix = np.hstack((std_matrix, volume_matrix))
    mean_std_volume_matrix = np.hstack((mean_matrix, std_matrix, volume_matrix))

    logger.info("Reading ROI labels from atlas text file.")
    index_ROI = []
    with open(atlas_txt, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) > 1:
                index_ROI.append(columns[1].strip())

    logger.debug(f"ROI labels found: {index_ROI}")

    logger.info("Generating feature column names for each ROI and statistic.")
    index_ROI_mean = [roi + '_mean' for roi in index_ROI]
    index_ROI_std = [roi + '_std' for roi in index_ROI]
    index_ROI_volume = [roi + '_volume' for roi in index_ROI]

    index_ROI_mean_std = index_ROI_mean + index_ROI_std
    index_ROI_mean_volume = index_ROI_mean + index_ROI_volume
    index_ROI_mean_std_volume = index_ROI_mean + index_ROI_std + index_ROI_volume
    index_ROI_std_volume = index_ROI_std + index_ROI_volume

    logger.info("Creating pandas DataFrames for each feature combination.")
    df_mean = pd.DataFrame(mean_matrix[:, data_start:],
                           index=range(std_matrix[:, data_start:].shape[0]),
                           columns=index_ROI)

    df_std = pd.DataFrame(std_matrix[:, data_start:],
                          index=range(std_matrix[:, data_start:].shape[0]),
                          columns=index_ROI)

    df_volume = pd.DataFrame(volume_matrix[:, data_start:],
                             index=range(std_matrix[:, data_start:].shape[0]),
                             columns=index_ROI)

    df_mean_std = pd.DataFrame(mean_std_matrix[:, data_start:],
                               index=range(std_matrix[:, data_start:].shape[0]),
                               columns=index_ROI_mean_std)

    df_mean_volume = pd.DataFrame(mean_volume_matrix[:, data_start:],
                                  index=range(std_matrix[:, data_start:].shape[0]),
                                  columns=index_ROI_mean_volume)

    df_mean_std_volume = pd.DataFrame(mean_std_volume_matrix[:, data_start:],
                                      index=range(std_matrix[:, data_start:].shape[0]),
                                      columns=index_ROI_mean_std_volume)

    df_std_volume = pd.DataFrame(std_volume_matrix[:, data_start:],
                                 index=range(std_matrix[:, data_start:].shape[0]),
                                 columns=index_ROI_std_volume)

    logger.success("Feature extraction completed successfully.")

    return df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume


def classify_independent_dataset(data_frame, model_path):
    """
    Classifies a new data sample using a previously trained machine learning model.

    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the features for classification.
        model_path (str): Path to the saved model file (joblib format).

    Returns:
        Tuple[int, np.ndarray]: Predicted class label and class probabilities.
    """
    
    logger.info(f"Loading trained model from: {model_path}")
    model = joblib.load(model_path)

    logger.info("Making prediction using pipeline.")
    classification = model.predict(data_frame)[0]
    probability = model.predict_proba(data_frame)[0]

    logger.success("Prediction completed.")
    
    return classification, probability

    
