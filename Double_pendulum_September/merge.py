import pandas as pd
import glob

# Lista dei file CSV da combinare
file_paths = glob.glob("*.csv")

# Leggi il primo file per inizializzare il dataframe
combined_df = pd.read_csv(file_paths[0])

# Loop attraverso gli altri file e aggiungi al dataframe combinato
for file_path in file_paths[1:]:
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Scrivi il dataframe combinato in un nuovo file CSV
combined_df.to_csv("combined_data.csv", index=False)

print("Combinazione completata. Il dataset combinato è stato salvato come 'combined_data.csv'.")