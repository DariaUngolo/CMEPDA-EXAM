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
    Extracts features (mean, standard deviation, volume) from a NIfTI image using a MATLAB function
    and organizes them into DataFrames. Additionally, it reads an atlas file and corresponding labels
    to identify Regions of Interest (ROIs).
    
    Parameters
    ----------
    nifti_image_path : str
        Path to the NIfTI image file to be processed.
        
    atlas_file : str
        Path to the atlas NIfTI file that defines regions for feature extraction.
        
    atlas_txt : str
        Path to the atlas labels text file containing region names or identifiers.
        
    matlab_feature_extractor_path : str
        Path to the MATLAB script or function used for feature extraction.
    
    Returns
    -------
    tuple
        A tuple of pandas DataFrames representing extracted features, organized as follows:
        
        - df_mean : pd.DataFrame
            Mean intensity values for each ROI defined by the atlas.
        
        - df_std : pd.DataFrame
            Standard deviation of intensity values for each ROI.
        
        - df_volume : pd.DataFrame
            Volume of each ROI as defined by the atlas.
        
        - df_mean_std : pd.DataFrame
            Combined features containing both mean and standard deviation for each ROI.
        
        - df_mean_volume : pd.DataFrame
            Combined features containing both mean and volume for each ROI.
        
        - df_std_volume : pd.DataFrame
            Combined features containing both standard deviation and volume for each ROI.
        
        - df_mean_std_volume : pd.DataFrame
            Combined features containing mean, standard deviation, and volume for each ROI.
    
    Notes
    -----
    - The NIfTI image and atlas must be aligned in the same space for accurate feature extraction.
    - The MATLAB function should be configured to read the image and atlas paths, perform computation,
      and output the results in a structured format.
    - Make sure to install the required Python-MATLAB interface package to enable interaction.
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

    
