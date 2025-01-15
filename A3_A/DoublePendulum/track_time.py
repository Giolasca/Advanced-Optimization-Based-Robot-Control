import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Path di base per ciascun tipo di controller
base_path_ltc = '/home/student/orc/RL/A3_new/SP/Plots_&_Animations'
base_path_full = '/home/student/orc/RL/A3_new/SP/Plots_&_Animations_NTC'

# Numero totale di simulazioni
num_simulations = 15

# Nome della colonna dei tempi di esecuzione
execution_time_column = 'Comp_Times'  # Assumendo che la colonna si chiami "Comp_Times"
nn_eval_time_column = 'NN_Eval_Times'  # Nome della colonna per il tempo di valutazione della rete neurale

# Funzioni per calcolare il tempo medio e worst-case
def compute_average_case(data):
    return data.mean()

def compute_worst_case(data):
    return data.max()

# Funzione per calcolare le statistiche da tutti i file di una simulazione
def calculate_statistics(base_path, prefix, exclude_tests=[], exclude_nn_eval=False):
    avg_times = []
    worst_times = []
    
    for i in range(1, num_simulations + 1):
        if i in exclude_tests:
            continue  # Skip the tests to be excluded
        
        # Costruzione del percorso del file
        file_path = os.path.join(base_path, f'Test_225_unconstr_tanh_test_{i}', f'mpc_results_225_unconstr_tanh_test_{i}.csv')
        
        # Caricamento del file CSV
        df = pd.read_csv(file_path)
        
        # Estrazione dei dati di esecuzione
        execution_times = df[execution_time_column]
        
        # Calcolo delle statistiche per la simulazione corrente
        avg_times.append(compute_average_case(execution_times))
        worst_times.append(compute_worst_case(execution_times))

        # Se richiesto, rimuovere il tempo di valutazione della rete neurale dall'avg e dal worst
        if exclude_nn_eval:
            nn_eval_time = df[nn_eval_time_column].mean() if nn_eval_time_column in df else 0
            avg_times[-1] -= nn_eval_time
            worst_eval_time = df[nn_eval_time_column].max() if nn_eval_time_column in df else 0
            worst_times[-1] -= worst_eval_time

    # Calcolo del tempo medio e worst-case totale su tutte le simulazioni
    total_avg_time = np.mean(avg_times) if avg_times else np.nan
    total_worst_time = np.max(worst_times) if worst_times else np.nan
    
    return total_avg_time, total_worst_time

# Calcolo delle statistiche per LTC-NMPC escludendo i test 10 e 14, rimuovendo il tempo NN
ltc_avg, ltc_worst = calculate_statistics(base_path_ltc, 'ltc_nmpc', exclude_tests=[10, 14], exclude_nn_eval=True)

# Calcolo delle statistiche per Full MPC escludendo i test 10 e 14
full_avg, full_worst = calculate_statistics(base_path_full, 'full_mpc', exclude_tests=[10, 14])

# Creazione del grafico a barre per il confronto
labels = ['Average-case', 'Worst-case']  # Cambiato l'ordine
ltc_values = [ltc_avg, ltc_worst]  # Cambiato l'ordine
full_values = [full_avg, full_worst]  # Cambiato l'ordine

x = np.arange(len(labels))  # etichette per le barre
width = 0.35  # larghezza delle barre

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ltc_values, width, label='MPC with terminal cost', color='blue')
rects2 = ax.bar(x + width/2, full_values, width, label='Full MPC', color='orange')

# Aggiungi etichette, titolo e personalizzazioni
ax.set_ylabel('CPU Time [ms]')
ax.set_title('Comparison of Execution Time')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Aggiunta dei valori sopra le barre
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()