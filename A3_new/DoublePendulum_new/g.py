import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Step 1: Leggi i dati dal file CSV
df = pd.read_csv('ocp_data_DP.csv')

# Visualizza le prime righe del DataFrame per verificare che i dati siano stati caricati correttamente
print(df.head())

# Step 2: Creazione di un pair plot con seaborn
sns.pairplot(df)
plt.show()

# Step 3: Creazione di una visualizzazione 3D interattiva con plotly
fig = px.scatter_3d(df, x='q1', y='v1', z='q2', color='v2', size='Costs', title='5D Data Visualization')
fig.show()


