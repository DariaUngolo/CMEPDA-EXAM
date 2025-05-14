import pandas as pd
from pathlib import Path
import sys
#import random_forest
import random_forest_PCA
import matlab.engine
from f_alternative_matlab_engine_NUOVOATLANTE import feature_extractor

# Import necessary modules
import argparse

if __name__ == '__main__':

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Add the current directory to the MATLAB path
    #eng.addpath(r'C:\Users\brand\OneDrive\Desktop\CMEPDA-EXAM', nargout=0)
    eng.addpath(r'C:\Users\daria\OneDrive\Desktop\CIAO\CMEPDA-EXAM', nargout=0)


    #Define file paths for input data and output files

    #folder_path = r"C:\Users\daria\OneDrive\Desktop\ESAME\tutti_i_dati"
    #atlas_file = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40.spm5.avg152T1.gm.label.nii.gz"
    #atlas_txt = r"C:\Users\daria\OneDrive\Desktop\ESAME\lpba40_labelID.txt"
    #output_csv_prefix = r"C:\Users\daria\OneDrive\Desktop\ESAME\outputpython"
    #metadata_csv = r"C:\Users\daria\OneDrive\Desktop\ESAME\AD_CTRL_metadata.csv"

    #folder_path = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\AD_CTRL"
    #atlas_file = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\BN_Atlas_246_2mm.nii.gz"
    #atlas_txt = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\BN_Atlas_246_LUT.txt"
   # output_csv_prefix = r"C:\Users\brand\OneDrive\Desktop\OutputValues"
    #metadata_csv = r"C:\Users\brand\OneDrive\Desktop\CMEPDA\progetto esame\data\AD_CTRL_metadata.csv"


    folder_path = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\tutti_i_dati"
    atlas_file = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\BN_Atlas_246_2mm.nii.gz"
    atlas_txt = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\BN_Atlas_246_LUT.txt"
    output_csv_prefix = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\outputpythonNUOVOATLANTE"
    metadata_csv = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\AD_CTRL_metadata.csv"





    # Load metadata and sort by ID
    df_group = pd.read_csv(metadata_csv, sep='\t')
    df_group_selected = df_group[["ID", "DXGROUP"]].sort_values(by="ID")
    group_1= df_group_selected[["DXGROUP"]]
    print(df_group.columns)
    print(df_group_selected)
    print(group_1)


    # Call feature extraction function
    df_mean, group, df_volume, df_VM  = feature_extractor(folder_path, atlas_file, atlas_txt, metadata_csv, output_csv_prefix)




    # Verifica dimensioni dei DataFrame
    print("Numero soggetti (feature):", df_mean.shape[0])
    print("Numero soggetti (etichette):", df_group_selected.shape[0])


    # Se non coincidono, stampa le differenze
    if df_mean.shape[0] != df_group_selected.shape[0]:
        print("⚠️ Attenzione: numero di soggetti non corrispondente tra feature e metadati.")

    # Supponiamo che df_mean abbia le righe nello stesso ordine degli ID
    subject_ids = df_group_selected["ID"].values  # array degli ID ordinati
    print(subject_ids)

    df_mean.index = subject_ids
    df_VM.index = subject_ids
    #df_std.index = subject_ids
    #df_unita.index = subject_ids
    group = df_group_selected.set_index("ID")["DXGROUP"]



   # print("dimensione di matrice unita")
   # print(df_unita.shape)

    print("dimensione di df_mean")
    print(df_mean.shape)

    print("dimensione di group")
    print(group.shape)

    #print("Indice di features combined:")
    #print(df_unita.index)
    print("Indice di group:")
    print(group.index)

    print("stampa group")
    print(group)

    print("stampa df_mean")
    #print(df_mean)

    #print("stampa  df_unita")
    #print( df_unita)

    # Evaluate the Random Forest classifier
    random_forest.RFPipeline_noPCA(df_VM, group, 10, 5)
    #random_forest_PCA.RFPipeline_PCA(df_VM, group,10, 5)
