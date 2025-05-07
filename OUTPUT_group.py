import numpy as np
import pandas as pd
import matlab.engine
import sys
from pathlib import Path
import os

# Add the parent directory to sys.path to allow imports from the parent folder
sys.path.insert(0, str(Path(os.getcwd()).parent))

def feature_extractor(folder_path, atlas_file, atlas_txt, output_csv_prefix, metadata_csv):
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

    eng.addpath(r"C:\Users\daria\OneDrive\Desktop\NUOVO_GIT\CMEPDA-EXAM", nargout=0)
    current_folder = eng.pwd()

    # Call a MATLAB function to extract mean and standard deviation of the features
    [mean, std] = eng.f_feature_extractor_means_stds(folder_path, atlas_file, atlas_txt, output_csv_prefix, nargout=2)
    eng.quit()

    # Convert the result arrays (mean and std) to numpy arrays
    mean_t = np.asarray(mean)
    std_t = np.asarray(std)

    # Create pandas DataFrame for mean and std values
    df_mean = pd.DataFrame(mean_t[1:, 1:], index=mean_t[0, 1:], columns=mean_t[1:, 0])
    df_std = pd.DataFrame(std_t[1:, 1:], index=std_t[0, 1:], columns=std_t[1:, 0])

    # Read the metadata CSV file that contains group information
    df_group = pd.read_csv(metadata_csv, sep='\t')

    # Corrected column access (removed quotes from column names)
    df_group_selected = df_group[["ID", "DXGROUP"]]
    group = df_group[["DXGROUP"]]

    return df_mean, df_std, group, df_group_selected

if __name__ == "__main__":

    # Define file paths for input data and output files
    folder_path = r"C:\Users\daria\OneDrive\Desktop\ESAME\tutti_i_dati"
    atlas_file = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40.spm5.avg152T1.gm.label.nii.gz"
    atlas_txt = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40_labelID.txt"
    output_csv_prefix = r"C:\Users\daria\OneDrive\Desktop\OUTPUT\outputpythoncambio"
    metadata_csv = r"C:\Users\daria\OneDrive\Desktop\ESAME\AD_CTRL_metadata.csv"
    output_path = r"C:\Users\daria\OneDrive\Desktop\OUTPUT"

    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # Call the feature extraction function and obtain the results
    df_mean, df_std, group, df_group_selected = feature_extractor(folder_path, atlas_file, atlas_txt, output_csv_prefix, metadata_csv)

    # Print the results for debugging
    print("=== DataFrame Selected ===")
    print(df_group_selected)
    print("=== Group ===")
    print(group)

    # Create and save CSV files
    file_name1 = "df_group_selected.csv"
    file_name2 = "group.csv"
    file_path1 = os.path.join(output_path, file_name1)
    file_path2 = os.path.join(output_path, file_name2)

    try:
        df_group_selected.to_csv(file_path1, index=False)
        group.to_csv(file_path2, index=False)
        print(f"File salvato correttamente in {file_path1} e {file_path2}")
    except Exception as e:
        print(f"Errore durante il salvataggio dei file: {e}")

    # Print the resulting dataframes
    print("=== DataFrame Mean ===")
    print(df_mean)
    print("\n=== DataFrame Std ===")
    print(df_std)
