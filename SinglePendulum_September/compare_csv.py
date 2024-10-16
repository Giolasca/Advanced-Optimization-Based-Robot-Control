import pandas as pd

def compara_costi(file1, file2):
    # Carica i dati dai file CSV
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Unisce i due DataFrame in base alle colonne 'position' e 'velocity'
    merged_df = pd.merge(df1, df2, on=['position', 'velocity'], suffixes=('_file1', '_file2'))
    
    # Seleziona solo le colonne dei costi da entrambi i file
    costi_df = merged_df[['position', 'velocity', 'cost_file1', 'cost_file2']]
    
    return costi_df

# Esempio di utilizzo
file1 = 'ocp_data_SP_target_180_constr.csv'
file2 = 'ocp_data_SP_target_180_unconstr.csv'
risultato = compara_costi(file1, file2)

# Stampa o salva i risultati
print(risultato)
