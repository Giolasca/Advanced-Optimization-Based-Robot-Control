import pandas as pd
import numpy as np

# Carica il file CSV
df = pd.read_csv('ocp.csv')

# Duplica le righe
df_doppio = pd.concat([df, df])

# Mescola le righe
df_mischiato = df_doppio.sample(frac=1).reset_index(drop=True)

# Salva il nuovo file CSV
df_mischiato.to_csv('ocp_data.csv', index=False)
