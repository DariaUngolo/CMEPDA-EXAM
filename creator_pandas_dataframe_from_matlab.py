import pandas as pd

def load_results_from_csv(mean_csv_file, std_csv_file):
    # Carica i file CSV in due DataFrame separati
    mean_df = pd.read_csv(mean_csv_file)
    std_df = pd.read_csv(std_csv_file)
    
    # Visualizza i DataFrame per verifica
    print("--- Mean Table ---")
    print(mean_df)
    print("--- Standard Deviation Table ---")
    print(std_df)
    
    return mean_df, std_df

# Esegui la funzione con i file CSV generati da MATLAB
mean_csv_file = "C:\\Users\\brand\\OneDrive\\Desktop\\CMEPDA EXAM\\CMEPDA-EXAM\\AD_results_mean.csv"
std_csv_file = "C:\\Users\\brand\\OneDrive\\Desktop\\CMEPDA EXAM\\CMEPDA-EXAM\\AD_results_std.csv"

mean_df, std_df = load_results_from_csv(mean_csv_file, std_csv_file)

print("Mean DataFrame:")
print(mean_df.head())  # Visualizza le prime righe del DataFrame medio