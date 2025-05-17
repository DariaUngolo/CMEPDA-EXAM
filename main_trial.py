import pandas as pd
from pathlib import Path
import sys
import classifiers_unified
import matlab.engine
from feature extractor import feature_extractor

# Import necessary modules
import argparse

if __name__ == '__main__':

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()



    # Call feature extraction function

    df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume, diagnostic_group_labels = feature_extractor(folder_path, atlas_file, atlas_txt, metadata_csv, output_csv_prefix)


    # Se non coincidono, stampa le differenze
    if df_mean.shape[0] != df_diagnostic_group_labels.shape[0]:
        print("⚠️ Attenzione: numero di soggetti non corrispondente tra feature e metadati.")

   

    # Evaluate the Random Forest classifier

    #random_forest.RFPipeline_noPCA(df_std_volume, group, 10, 5)
    #random_forest_PCA.RFPipeline_PCA(df_unita, group, 10, 5)
    #random_forest_RFECV.RFPipeline_RFECV(df_std_volume, group, 10, 5)
    #SVM_simple.SVM_simple(df_unita, group, "rbf")

