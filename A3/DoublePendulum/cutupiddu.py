import pandas as pd

# Carica il tuo file CSV con il separatore tab
input_file_path = 'data.csv'
output_file_path = 'data_modified.csv'

# Leggi il file CSV con il separatore virgola
df = pd.read_csv(input_file_path, sep='\t')

# Salva il DataFrame risultante in un nuovo file CSV con il separatore virgola
df.to_csv(output_file_path, index=False, sep=',')

print("Il file è stato generato con successo.")