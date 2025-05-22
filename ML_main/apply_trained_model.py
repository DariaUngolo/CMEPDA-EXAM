import argparse
import numpy as np
import pandas as pd
import joblib  # Per caricare il modello salvato
import matlab.engine
from pathlib import Path
import os

def feature_extractor_independent_dataset(nifti_image_path, atlas_file, atlas_txt, matlab_feature_extractor_path):


    # === 1. Start the MATLAB engine ===
    eng = matlab.engine.start_matlab()

    # Add the MATLAB path to the engine
    eng.addpath(matlab_feature_extractor_path, nargout=0)

    # === 2. Call the MATLAB function to extract features ===

    mean, std, volume = eng.feature_extractor(nifti_image_path, atlas_file, atlas_txt, nargout=3)


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


    df_mean = pd.DataFrame(mean_matrix[:, data_start:],
                        index=range(std_matrix[:, data_start:].shape[0]),
                        columns=index_ROI)


    df_std = pd.DataFrame(std_matrix[:, data_start :],
                        index=range(std_matrix[:, data_start:].shape[0]),
                        columns=index_ROI)

    df_volume = pd.DataFrame(volume_matrix[:, data_start :],
                        index=range(std_matrix[:, data_start:].shape[0]),
                        columns=index_ROI)

    df_mean_std= pd.DataFrame(mean_std_matrix[:, data_start :],
                        index=range(std_matrix[:, data_start:].shape[0]),
                        columns=index_ROI_mean_std)

    df_mean_volume = pd.DataFrame(mean_volume_matrix[:, data_start :],
                        index=range(std_matrix[:, data_start:].shape[0]),
                        columns=index_ROI_mean_volume)

    df_mean_std_volume = pd.DataFrame(mean_std_volume_matrix[:, data_start :],
                           index=range(std_matrix[:, data_start:].shape[0]),
                            columns=index_ROI_mean_std_volume)

    df_std_volume = pd.DataFrame(std_volume_matrix[:, data_start :],
                            index=range(std_matrix[:, data_start:].shape[0]),
                            columns=index_ROI_std_volume)

       return df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume



def classify_image(data_frame, model_path):


    # Predizione con il modello
    model = joblib.load(model_path)
    classification = model.predict(data_frame)[0]
    probability = model.predict_proba(data_frame)[0]


    return classification, probability

# Main con argparse per NIfTI
if __name__ == "__main__":




    # Percorsi predefiniti per gli altri file
    NIFTI_FILE =
    ATLAS_FILE =
    ATLAS_TXT =
    METADATA_CSV =
    MATLAB_FEATURE_EXTRACTOR_PATH =
    MODEL_PATH =

    try:
        predicted_class = classify_image(
            NIFTI_FILE,
            ATLAS_FILE,
            ATLAS_TXT,
            METADATA_CSV,
            MATLAB_FEATURE_EXTRACTOR_PATH,
            MODEL_PATH
        )

        print(f"L'immagine appartiene alla classe: {predicted_class}")
    except Exception as e:
        print(f"Errore durante la classificazione: {e}")
