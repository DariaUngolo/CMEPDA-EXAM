import numpy as np
import pandas as pd
import matlab.engine
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))


def feature_extractor(image_filepaths,atlas_file, atlas_txt):
    """
    Uses the MATLAB Engine API to run the feature_extractor.m function. From the outputs of that function, it defines 2
    dataframes containing the extracted features and a series containing the labels of the respective subjects.

    Parameters
    ----------
        image_filepaths : list
            Paths to the diffusion parameters maps.
        masks_filepaths : list
            Paths to the diffusion space segmentations.

    Returns
    -------
        df_mean : pandas.DataFrame
            Mean of pixel values for each region (columns) and each subject (rows).
        df_std : pandas.DataFrame
            Standard deviation of pixel values for each region (columns) and each subject (rows).
        group : pandas.Series
            Subject labels.


        atlas_file, atlas_txt
    """



"""
        PARTE STANDARD PER RICHIAMARE MATLAB, CERCA DI CAPIRE SE è L'UNICO MODO

"""
# Start MATLAB engine
    eng = matlab.engine.start_matlab()

    eng.addpath('./ML_tools')

    current_folder=(eng.pwd())

# Call a MATLAB function
    [mean, std] = eng.feature_extractor(image_filepaths, atlas_file, atlas_txt, nargout=2)  #richiama la funzione di matlab
# Stop MATLAB engine
    eng.quit()









# Create Pd dataframe

    mean_t = np.asarray(mean)  #converte la matrice in array di python
    std_t = np.asarray(std)
    n_regxsub = np.shape(mean_t[1])    #determina forma della matrice restituita dalla funzione matlab


  #da capire se ci serve fare il trasposto


    df_mean = pd.DataFrame(mean_t[1:, 1:],   #Questo indica che si stanno prendendo tutte le righe di mean_t escludendo la prima riga dove ci sono gli indici delle roi e dalla seconda colonna in poi uguale dato che nell prima colonna ci sono i nomi dei soggetti
                           index=mean_t[0, 1:]   #index: Definisce le etichette (nomi) delle righe nel DataFrame.
                           columns=mean_t[1:, 0])

    df_std = pd.DataFrame(std_t[1:, 1:(n_regxsub[0]-1)],
                           index=mean_t[0, 1:]   #index: Definisce le etichette (nomi) delle righe nel DataFrame.
                           columns=mean_t[1:, 0])


    """
            Questo crea un DataFrame df_mean dove:
            Le righe sono i soggetti (estratti da mean[0]).
            Le colonne sono le regioni (estratte da region).
            I valori sono i valori medi dei pixel per ciascun soggetto e regione.
          a sintassi mean_t[1:, 1:(n_regxsub[0]-1)] indica come vengono selezionati i dati da mean_t:

    """

    df_group = pd.read_csv('AD_CTRL_metadata.csv')  #legge file csv con dati dei soggetti
    #df_group.sort_values(by=["ID"], inplace=True)   #NON CAPISCO PERCHè FA QUESTA COSA DI ORDINAMENTO
    group = df_group["DXGROUP"]


    #QUI ORDINA IL FILE IN CUI SAPPIAMO SE SOGGETTO è MALATO O NO


    return df_mean, df_std, group
