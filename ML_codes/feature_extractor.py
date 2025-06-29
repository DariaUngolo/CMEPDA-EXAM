import numpy as np
import pandas as pd
import matlab.engine
import sys
from pathlib import Path
import os

# Add the parent directory to sys.path to allow imports from the parent folder
sys.path.insert(0, str(Path(os.getcwd()).parent))

def feature_extractor(folder_path, atlas_file, atlas_txt, metadata_csv, matlab_feature_extractor_path):

    """

    Extract feature means and standard deviations from brain image data using a MATLAB function.

    This function interfaces with MATLAB to extract mean and standard deviation values from image data in a specified folder,
    based on a given atlas. It processes MATLAB outputs into NumPy arrays, organizes them into pandas DataFrames,
    and loads metadata about the image groups. The function returns the processed feature data and group labels.

    Parameters
    ----------
    folder_path : str  
        Path to the folder containing the image data.
    
    atlas_file : str  
        Path to the atlas file used for feature extraction.
    
    atlas_txt : str  
        Path to the text file containing atlas-related information.
    
    metadata_csv : str  
        Path to the CSV file containing metadata about the image groups.
    
    matlab_feature_extractor_path : str  
        Path to the MATLAB function used for feature extraction.
    
    Returns
    -------
    df_mean : pandas.DataFrame  
        DataFrame with mean values of features.
    
    df_std : pandas.DataFrame  
        DataFrame with standard deviation values of features.
    
    group : pandas.Series  
        Series containing group labels from the metadata.
    
    df_unita : pandas.DataFrame  
        DataFrame with combined mean and standard deviation values.
    
    df_media_volume : pandas.DataFrame  
        DataFrame with mean values and volume.
    
    df_media_std_volume : pandas.DataFrame  
        DataFrame with mean, standard deviation, and volume.
    
    df_std_volume : pandas.DataFrame  
        DataFrame with standard deviation and volume values.



    Notes
    -----
    - The function requires MATLAB integration and appropriate MATLAB paths.
    - Outputs from MATLAB are converted to NumPy arrays before creating pandas DataFrames.
    - Group labels are loaded from the provided metadata CSV file.


    References
    ----------
    - https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    - https://pandas.pydata.org/


    """



    # === 1. Start the MATLAB engine ===
    eng = matlab.engine.start_matlab()

    # Add the MATLAB path to the engine
    eng.addpath(matlab_feature_extractor_path, nargout=0)

    # === 2. Call the MATLAB function to extract features ===

    mean, std, volume = eng.feature_extractor(folder_path, atlas_file, atlas_txt, nargout=3)


    # Quit MATLAB engine after the operation is complete
    eng.quit()

    # === 3. Convert MATLAB arrays to NumPy arrays ===
    mean_matrix = np.asarray(mean)
    std_matrix = np.asarray(std)
    volume_matrix = np.asarray(volume)


    # === 4. Check if the first row contains headers or data ===
    data_start = 0
    if isinstance(mean_matrix[0, 0], str) :
        data_start = 1  # First row contains headers

    if isinstance(std_matrix[0, 0], str) :
        data_start = 1  # First row contains headers

    if isinstance(volume_matrix[0, 0], str) :
        data_start = 1  # First row contains headers

    # === 5. Create matrices for different combinations of features ===

    mean_std_matrix = np.hstack((mean_matrix, std_matrix))
    mean_volume_matrix = np.hstack((mean_matrix, volume_matrix))
    std_volume_matrix = np.hstack((std_matrix, volume_matrix))
    mean_std_volume_matrix = np.hstack((mean_matrix, std_matrix, volume_matrix))


    # === 6. Create pandas DataFrames for features values ===

    # Create a list of ROI names from the atlas text file
    index_ROI = []
    with open(atlas_txt, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) > 1:  # Ensure there are at least two columns because the first is the index and the second is the name of the ROI
                index_ROI.append(columns[1].strip()) # strip() removes any leading/trailing whitespace

    #  Need for distinct names for each statistics of each ROI
    index_ROI_mean = [roi + '_mean' for roi in index_ROI]
    index_ROI_std = [roi + '_std' for roi in index_ROI]
    index_ROI_volume = [roi + '_volume' for roi in index_ROI]

    # Combine the names for the DataFrame columns
    index_ROI_mean_std = index_ROI_mean + index_ROI_std
    index_ROI_mean_volume = index_ROI_mean + index_ROI_volume
    index_ROI_mean_std_volume = index_ROI_mean + index_ROI_std + index_ROI_volume
    index_ROI_std_volume = index_ROI_std + index_ROI_volume



    # === 7. Load metadata from CSV file ===
    df_dxgroup = pd.read_csv(metadata_csv, sep='\t')
    df_dxgroup.sort_values(by=[df_dxgroup.columns[0]], inplace=True)

    # === 8. Extract group labels (metadata) ===
    diagnostic_group_labels = df_dxgroup.iloc[:, 1]
    diagnostic_group_labels.index = df_dxgroup.iloc[:, 0]  # Set the index to the subject ID
    
    label_id_row =  df_dxgroup.iloc[:, 0].values # First column is the subject ID


    # Create DataFrames for the mean, std, and volume arrays
    if len(index_ROI) != mean_matrix[:, data_start:].shape[1]:
        raise ValueError("Mismatch between index_ROI and number of columns in mean_array slice.")

    df_mean = pd.DataFrame(mean_matrix[:, data_start:],
                        index=label_id_row,  # First column is the subject ID
                        columns=index_ROI)


    df_std = pd.DataFrame(std_matrix[:, data_start :],
                        index=label_id_row,
                        columns=index_ROI)

    df_volume = pd.DataFrame(volume_matrix[:, data_start :],
                        index=label_id_row,
                        columns=index_ROI)

    df_mean_std= pd.DataFrame(mean_std_matrix[:, data_start :],
                        index=label_id_row,
                        columns=index_ROI_mean_std)

    df_mean_volume = pd.DataFrame(mean_volume_matrix[:, data_start :],
                        index=label_id_row,
                        columns=index_ROI_mean_volume)

    df_mean_std_volume = pd.DataFrame(mean_std_volume_matrix[:, data_start :],
                            index=label_id_row,
                            columns=index_ROI_mean_std_volume)

    df_std_volume = pd.DataFrame(std_volume_matrix[:, data_start :],
                            index=label_id_row,
                            columns=index_ROI_std_volume)



    # === 9. Return results ===
    return df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume, diagnostic_group_labels

