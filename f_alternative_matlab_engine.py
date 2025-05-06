import numpy as np
import pandas as pd
import matlab.engine
import sys
from pathlib import Path
import os

# Add the parent directory to sys.path to allow imports from the parent folder
sys.path.insert(0, str(Path(os.getcwd()).parent))



def feature_extractor(folder_path, atlas_file, atlas_txt, output_csv_prefix):
    """
    Extracts features from MRI data using a MATLAB-based feature extraction function.

    Args:
    - folder_path (str): Path to the folder containing the MRI data files.
    - atlas_file (str): Path to the brain atlas file in NIfTI format.
    - atlas_txt (str): Path to the text file that contains atlas information.
    - output_csv_prefix (str): Prefix for the output CSV files where results will be stored.

    Returns:
    - df_mean (DataFrame): A pandas DataFrame containing the mean values of the extracted features.
    - df_std (DataFrame): A pandas DataFrame containing the standard deviation values of the extracted features.
    - group (Series): A pandas Series containing the group labels (e.g., diagnosis group).
    """

# Start MATLAB engine
    eng = matlab.engine.start_matlab()

    eng.addpath(r"C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM", nargout=0)  # Add the current directory to MATLAB path
# Get the current MATLAB working directory
    current_folder=(eng.pwd())

# Call a MATLAB function to extract mean and standard deviation of the features
    # The function returns a tuple of (mean, std) values from the feature extraction process
    [ mean, std] = eng.f_feature_extractor_means_stds(folder_path, atlas_file, atlas_txt, output_csv_prefix, nargout=2)
# Stop MATLAB engine fter extraction is complete
    eng.quit()
# Convert the result arrays (mean and std) to numpy arrays for easier manipulation
    n_regxsub = np.shape(mean[:][1])
    mean_t = np.asarray(mean)
    std_t = np.asarray(std)

# Create a pandas DataFrame for mean values, skipping the first row and column for proper indexing
    df_mean = pd.DataFrame(mean_t[1:, 1:],
                           index=mean_t[0, 1:],   # Set the row labels as the second column of mean_t
                           columns=mean_t[1:, 0])  # Set the column labels as the first column of mean_t

# Create a pandas DataFrame for standard deviation values, similarly processing the std array
    df_std = pd.DataFrame(std_t[1:, 1:],
                           index=std_t[0, 1:],    # Set the row labels as the second column of std_t
                           columns=std_t[1:, 0])   # Set the column labels as the first column of std_t

# Read the metadata CSV file that contains group information (e.g., diagnosis)

    df_group = pd.read_csv(r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\AD_CTRL_metadata_AD.csv")
    df_group.sort_values(by=["ID"], inplace=True)   # Sort the dataframe by the "ID" column
    # Extract the diagnosis group labels (e.g., AD or CTRL) into a pandas Series
    group = df_group["DXGROUP"]

    return df_mean, df_std, group


