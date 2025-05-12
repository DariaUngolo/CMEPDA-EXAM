import pandas as pd
from pathlib import Path
import sys
import random_forest
import matlab.engine
from f_alternative_matlab_engine import feature_extractor

# Import necessary modules
import argparse

if __name__ == '__main__':

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Add the current directory to the MATLAB path
    #eng.addpath(r'C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM', nargout=0)
    eng.addpath(r'C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM', nargout=0)

    # Define file paths for input data and output files
    #folder_path = r"C:\Users\daria\OneDrive\Desktop\ESAME\tutti_i_dati"
    #atlas_file = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40.spm5.avg152T1.gm.label.nii.gz"
    #atlas_txt = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40_labelID.txt"
    #output_csv_prefix = r"C:\Users\daria\OneDrive\Desktop\ESAME\outputpython"
    #metadata_csv = r"C:\Users\daria\OneDrive\Desktop\ESAME\AD_CTRL_metadata.csv"

    folder_path = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\AD_CTRL"
    atlas_file = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\lpba40.spm5.avg152T1.gm.label.nii.gz"
    atlas_txt = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\lpba40_labelID.txt"
    output_csv_prefix = r"C:\Users\brand\OneDrive\Desktop\outputpython"
    metadata_csv = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\AD_CTRL_metadata.csv"



    # Load metadata and sort by ID
    df_group = pd.read_csv(metadata_csv, sep='\t')
    df_group_selected = df_group[["ID", "DXGROUP"]].sort_values(by="ID")
    group_1= df_group_selected[["DXGROUP"]]
    print(df_group.columns)
    print(df_group_selected)
    print(group_1)


    # Call feature extraction function
    df_mean, df_std, group = feature_extractor(folder_path, atlas_file, atlas_txt, metadata_csv, output_csv_prefix)

    #combina il DataFrame df_mean e df_std
    df_mean.columns = [col + "_mean" for col in df_mean.columns]
    df_std.columns = [col + "_std" for col in df_std.columns]
    df_features_combined = pd.concat([df_mean, df_std], axis=1)


    # Verifica dimensioni dei DataFrame
    print("Numero soggetti (feature):", df_mean.shape[0])
    print("Numero soggetti (etichette):", df_group_selected.shape[0])
    print("df_features_combined shape:", df_features_combined.shape)  # dovrebbe essere (333, 2N)

    # Se non coincidono, stampa le differenze
    if df_mean.shape[0] != df_group_selected.shape[0]:
        print("⚠️ Attenzione: numero di soggetti non corrispondente tra feature e metadati.")

    # Supponiamo che df_mean abbia le righe nello stesso ordine degli ID
    subject_ids = df_group_selected["ID"].values  # array degli ID ordinati
    print(subject_ids)

    df_mean.index = subject_ids
    df_std.index = subject_ids
    df_features_combined.index = subject_ids
    group = df_group_selected.set_index("ID")["DXGROUP"]




    print("dimensione di df_mean")
    print(df_mean.shape)

    print("dimensione di group")
    print(group.shape)

    print("Indice di features combined:")
    print(df_features_combined.index)
    print("Indice di group:")
    print(group.index)

    print("stampa group")
    #print(group)
    print("stampa df_mean")
    #print(df_mean)

    # Evaluate the Random Forest classifier
    #random_forest.RFPipeline_noPCA(df_features_combined, group, 10, 5)
