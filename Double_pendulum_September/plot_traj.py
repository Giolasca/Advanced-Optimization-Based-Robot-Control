import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Funzione per leggere i dati da un file CSV
def read_csv_data(file_path):
    return pd.read_csv(file_path)

# Funzione per plottare i dati in subplot
def plot_trajectories(file1, file2):
    # Leggi i file CSV
    data1 = read_csv_data(file1)
    data2 = read_csv_data(file2)
    
    # Crea la figura e i subplot con layout 3x2
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # Primo subplot (q1_file1 e q1_file2)
    q1_ref = np.pi
    axs[0, 0].plot(data1['q1'], label='q1 with T = 0.01 & N = 5', color='b')
    axs[0, 0].plot(data2['q1'], label='q1 with T = 1 & N = 50', color='r')
    axs[0, 0].axhline(y=q1_ref, color='g', linestyle='--', label='q1_ref = π')
    axs[0, 0].set_title('q1 trajectory compare')
    axs[0, 0].set_xlabel('Time step')
    axs[0, 0].set_ylabel('q1 [rad]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Secondo subplot (q2_file1 e q2_file2)
    q2_ref = np.pi
    axs[0, 1].plot(data1['q2'], label='q2 with T = 0.01 & N = 5', color='b')
    axs[0, 1].plot(data2['q2'], label='q2 with T = 1 & N = 50', color='r')
    axs[0, 1].axhline(y=q2_ref, color='g', linestyle='--', label='q2_ref = π')
    axs[0, 1].set_title('q2 trajectory compare')
    axs[0, 1].set_xlabel('Time step')
    axs[0, 1].set_ylabel('q2 [rad]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Terzo subplot (v1_file1 e v1_file2)
    v_ref = 0
    axs[1, 0].plot(data1['v1'], label='v1 with T = 0.01 & N = 5', color='b')
    axs[1, 0].plot(data2['v1'], label='v1 with T = 1 & N = 50', color='r')
    axs[1, 0].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0')
    axs[1, 0].set_title('v1 trajectory compare')
    axs[1, 0].set_xlabel('Time step')
    axs[1, 0].set_ylabel('v1 [rad/s]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Quarto subplot (v2_file1 e v2_file2)
    axs[1, 1].plot(data1['v2'], label='v2 with T = 0.01 & N = 5', color='b')
    axs[1, 1].plot(data2['v2'], label='v2 with T = 1 & N = 50', color='r')
    axs[1, 0].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0')
    axs[1, 1].set_title('v2 trajectory compare')
    axs[1, 1].set_xlabel('Time step')
    axs[1, 1].set_ylabel('v2 [rad/s]')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Quinto subplot (u1_file1 e u1_file2)
    axs[2, 0].plot(data1['u1'], label='u1 with T = 0.01 & N = 5', color='b')
    axs[2, 0].plot(data2['u1'], label='u1 with T = 1 & N = 50', color='r')
    axs[2, 0].set_title('u1 trajectory compare')
    axs[2, 0].set_xlabel('Time step')
    axs[2, 0].set_ylabel('u1 [N/m]')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # Sesto subplot (u2_file1 e u2_file2)
    axs[2, 1].plot(data1['u2'], label='u2 with T = 0.01 & N = 5', color='b')
    axs[2, 1].plot(data2['u2'], label='u2 with T = 1 & N = 50', color='r')
    axs[2, 1].set_title('u2 trajectory compare')
    axs[2, 1].set_xlabel('Time step')
    axs[2, 1].set_ylabel('u2 [N/m]')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    # Aggiungi un layout pulito
    plt.tight_layout()
    plt.show()

# Specifica i percorsi dei file CSV
file1 = 'Plots_&_Animations/MPC_DoublePendulum_TC.csv'
file2 = 'Plots_&_Animations/MPC_DoublePendulum_NTC_T_1.csv'

# Esegui il plotting
plot_trajectories(file1, file2)
