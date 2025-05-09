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
    #eng.addpath(r"C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM", nargout=0)
    eng.addpath(r"C:\Users\daria\OneDrive\Desktop\CMEPDA-EXAM", nargout=0)

    # Get the current MATLAB working directory
    current_folder = eng.pwd()

    # === 2. Call the MATLAB function to extract features ===
    mean, std = eng.f_feature_extractor_means_stds(folder_path, atlas_file, atlas_txt, output_csv_prefix, nargout=2)

    # Quit MATLAB engine after the operation is complete
    eng.quit()

    # === 3. Convert MATLAB arrays to NumPy arrays ===
    mean_array = np.asarray(mean)
    std_array = np.asarray(std)

    # === 4. Transpose arrays if necessary (maintaining correct shape) ===
    if mean_array.shape[0] < mean_array.shape[1]:
        mean_array = mean_array.T
    if std_array.shape[0] < std_array.shape[1]:
        std_array = std_array.T

    # === 5. Check if the first row contains headers or data ===
    data_start = 0
    if isinstance(mean_array[0, 0], str) and mean_array[0, 0].lower() in ["image", "id", ""]:
        data_start = 1  # First row contains headers

    if isinstance(std_array[0, 0], str) and std_array[0, 0].lower() in ["image", "id", ""]:
        data_start = 1  # First row contains headers

    # === 6. Create pandas DataFrames for mean and standard deviation values ===
    df_mean = pd.DataFrame(mean_array[data_start:, 1:],
                           index=mean_array[data_start:, 0],
                           columns=mean_array[0, 1:])

    df_std = pd.DataFrame(std_array[data_start:, 1:],
                          index=std_array[data_start:, 0],
                          columns=std_array[0, 1:])

    # === 7. Load metadata from CSV file ===
    df_group = pd.read_csv(metadata_csv, sep='\t')
    df_group.sort_values(by=[df_group.columns[0]], inplace=True)

    # === 8. Extract group labels (metadata) ===
    group = df_group.iloc[:, 1]
    #    # Select only the first two columns (ID and Group)
    #group2 = df_group.iloc[:, [0, 1]]  # Columns with ID and Group

    # === 9. Return results ===
    return df_mean, df_std, group



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
