import pandas as pd

# 1. Carica il file CSV
file_path = 'combined_data_180_180.csv'  # Specifica il percorso del tuo file CSV
data = pd.read_csv(file_path)

# 2. Filtra i valori negativi della colonna 'cost'
negative_costs = data[data['cost'] < 0]

# 3. Controlla se ci sono valori negativi e stampali
if negative_costs.empty:
    print("Non ci sono valori negativi nella colonna 'cost'.")
else:
    print("Valori negativi trovati nella colonna 'cost':")
    print(negative_costs)

# Se desideri solo la lista dei valori negativi
negative_cost_values = negative_costs['cost'].tolist()
print("Valori negativi della colonna 'cost':", negative_cost_values)