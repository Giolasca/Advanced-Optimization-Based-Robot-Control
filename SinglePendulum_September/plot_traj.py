import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_trajectory_from_csv(filename):
    """
    Carica la traiettoria (posizione, velocità e input) da un file CSV.
    """
    return pd.read_csv(filename)

def plot_trajectories_with_reference(file1, file2, reference_position, reference_velocity, reference_input):
    """
    Carica le traiettorie da due file CSV e le plottano su un unico grafico, insieme ai valori di riferimento.
    """
    # Carica i dati dai file CSV
    traj1 = load_trajectory_from_csv(file1)
    traj2 = load_trajectory_from_csv(file2)
    
    # Creazione della figura
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot delle posizioni
    axs[0].plot(traj1['Positions'], label=f"Positions - {file1}", color='b')
    axs[0].scatter(range(len(traj2['Positions'])), traj2['Positions'], label=f"Positions - {file2}", color='g')
    axs[0].plot(reference_position, label="Reference", color='r', linestyle='--')  # Linea tratteggiata per il riferimento
    axs[0].set_title('Position Comparison')
    axs[0].set_ylabel('Position')
    axs[0].legend(loc='best')
    axs[0].grid(True)
    
    # Plot delle velocità
    axs[1].plot(traj1['Velocities'], label=f"Velocities - {file1}", color='b')
    axs[1].scatter(range(len(traj2['Velocities'])), traj2['Velocities'], label=f"Velocities - {file2}", color='g')
    axs[1].plot(reference_velocity, label="Reference", color='r', linestyle='--')  # Linea tratteggiata per il riferimento
    axs[1].set_title('Velocity Comparison')
    axs[1].set_ylabel('Velocity')
    axs[1].legend(loc='best')
    axs[1].grid(True)
    
    # Plot degli input
    axs[2].plot(traj1['Inputs'], label=f"Inputs - {file1}", color='b')
    axs[2].scatter(range(len(traj2['Inputs'])), traj2['Inputs'], label=f"Inputs - {file2}", color='g')
    axs[2].set_title('Input Comparison')
    axs[2].set_ylabel('Input')
    axs[2].set_xlabel('Time Steps')
    axs[2].legend(loc='best')
    axs[2].grid(True)

    # Aggiustare il layout
    plt.tight_layout()
    
    # Mostra il plot
    plt.show()

if __name__ == "__main__":
    # Specifica i file CSV
    csv_file1 = 'Plots_&_Animations/mpc_SP_NTC_135_unconstr.csv'
    csv_file2 = 'Plots_&_Animations/mpc_SP_TC_135_unconstr.csv'
    
    # Definisci il riferimento (valori costanti o variabili)
    reference_value = (3/4) * np.pi  # Costante di riferimento per la posizione
    time_steps = 50  # Numero di passi temporali
    
    reference_position = [reference_value] * time_steps  # Posizione di riferimento costante
    reference_velocity = [0] * time_steps  # Velocità di riferimento costante
    reference_input = [0] * time_steps     # Input di controllo costante
    
    # Esegui il plot con i riferimenti
    plot_trajectories_with_reference(csv_file1, csv_file2, reference_position, reference_velocity, reference_input)
