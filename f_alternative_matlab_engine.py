import numpy as np
import pandas as pd
import matlab.engine
import sys
from pathlib import Path
import os

# Add the parent directory to sys.path to allow imports from the parent folder
sys.path.insert(0, str(Path(os.getcwd()).parent))


def feature_extractor(folder_path, atlas_file, atlas_txt, metadata_csv, output_csv_prefix):
    """
    Extracts features from MRI images using a MATLAB-based feature extraction function.

    Args:
    - folder_path (str): Path to the folder containing the MRI image files (in NIfTI format).
    - atlas_file (str): Path to the brain atlas file in NIfTI format.
    - atlas_txt (str): Path to the text file that contains atlas information (ROI).
    - metadata_csv (str): Path to the CSV file containing metadata, including group information (e.g., diagnosis).
    - output_csv_prefix (str): Prefix for the output CSV files where results will be stored.

    Returns:
    - df_mean (DataFrame): A pandas DataFrame containing the mean values of the extracted features.
    - df_std (DataFrame): A pandas DataFrame containing the standard deviation values of the extracted features.
    - group (Series): A pandas Series containing the group labels (e.g., diagnosis group).
    """

    # === 1. Start the MATLAB engine ===
    eng = matlab.engine.start_matlab()

    # Add the MATLAB path (modify as needed)
    eng.addpath(r"C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM", nargout=0)

    # Get the current MATLAB working directory
    current_folder = eng.pwd()

    # === 2. Extract mean and standard deviation from the images using the MATLAB function ===
    # Call the MATLAB function to extract features (mean and std)
    [mean, std] = eng.f_feature_extractor_means_stds(folder_path, atlas_file, atlas_txt, output_csv_prefix, nargout=2)

    # After extraction, close the MATLAB engine
    eng.quit()

    # === 3. Convert the results into numpy arrays for easier manipulation ===
    mean_t = np.asarray(mean)
    std_t = np.asarray(std)

    # === 4. Transpose if necessary to maintain the correct shape of the data ===
    # Check if the shape of the data is incorrect; if so, transpose the matrices
    if mean_t[1:, 1:].shape != (len(mean_t[1:, 0]), len(mean_t[0, 1:])):
        mean_t = mean_t.T  # Transpose the mean matrix

    if std_t[1:, 1:].shape != (len(std_t[1:, 0]), len(std_t[0, 1:])):
        std_t = std_t.T  # Transpose the std matrix

    # === 5. Create a DataFrame for the mean values ===
    # Exclude the first row and first column (header and ID) to get the numeric data
    df_mean = pd.DataFrame(mean_t[1:, 1:], 
                           index=mean_t[1:, 0],  # Use the IDs or image names as the index
                           columns=mean_t[0, 1:])  # Use ROI names as the columns

    # === 6. Create a DataFrame for the standard deviation values ===
    df_std = pd.DataFrame(std_t[1:, 1:], 
                          index=std_t[1:, 0],  # Use the IDs or image names as the index
                          columns=std_t[0, 1:])  # Use ROI names as the columns

    # === 7. Read the metadata CSV file containing group information (e.g., diagnosis) ===
    df_group = pd.read_csv(metadata_csv, sep='\t')

    # Sort the data by the first column (ID or image name)
    df_group.sort_values(by=[df_group.columns[0]], inplace=True)

    # === 8. Extract the group labels (e.g., diagnosis: AD, CTRL, etc.) ===
    # Select only the first two columns (ID and Group)
    group2 = df_group.iloc[:, [0, 1]]  # Columns with ID and Group
    group1 = df_group.iloc[:, 1]  # Group labels (e.g., AD, CTRL)

    # === 9. Return the results ===
    # Return the DataFrames for the mean and standard deviation, and the group labels
    return df_mean, df_std, group1


    #return df_mean, df_std, group2 , group1, mean_t

#MAIN PER FARLO FUNZIONARE, GROUP2 è IL DATAFRAME CON LA COLONNA ID E COLONNA CON AD/Normal
#GROUP1 HA SOLO LABEL AD/NORMAL
#SE VUOI FAR FUNZIONARE IL CODICE SOTTO DEVI METTERE
#
#if __name__ == "__main__":

    #    QUESTO PROGRAMMA NON FA LA COLONNA GROUP

     #   Main script to run the feature extraction process and display the results.


        # Define the file paths for input data and output files

#    folder_path = r"C:\Users\daria\OneDrive\Desktop\ESAME\tutti_i_dati"
#    atlas_file = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40.spm5.avg152T1.gm.label.nii.gz"
#    atlas_txt = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40_labelID.txt"
#    output_csv_prefix = r"C:\Users\daria\OneDrive\Desktop\ESAME\outputpythonTRASPOSTO"
#    metadata_csv = r"C:\Users\daria\OneDrive\Desktop\ESAME\AD_CTRL_metadata.csv"


        # Call the feature extraction function and obtain the results

#    df_mean, df_std, group2, group, mean_t= feature_extractor(folder_path, atlas_file, atlas_txt, output_csv_prefix)

 #   print("Shape of mean_t[1:, 1:]:", mean_t[1:, 1:].shape)
 #   print("Length of mean_t[1:, 0]:", len(mean_t[1:, 0]))
 #   print("Length of mean_t[0, 1:]:", len(mean_t[0, 1:]))

 #   print(group)
        #Print the resulting dataframes and the group labels

  #  print("=== DataFrame Mean ===")
  #  print(df_mean)
  #  print("\n=== DataFrame Std ===")
  #  print(df_std)

        #print("\n===ID, Group ===")
        #print(group_selected)
        #print("\n===Group ===")
        #print(group)
