import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_trajectories(output_dir, num_tests):
    """
    Legge le traiettorie da ogni test e le plotta su un singolo grafico.
    
    :param output_dir: Directory dove sono salvati i risultati di ciascun test.
    :param num_tests: Numero totale dei test eseguiti.
    """
    
    # Inizializza le liste per raccogliere le traiettorie
    all_positions = []
    all_velocities = []
    all_inputs = []
    all_terminal_costs = []
    all_total_costs = []

    # Loop sui file generati per ciascun test
    for test_idx in range(1, num_tests + 1):
        # Trova la directory di ogni test
        test_dir = os.path.join(output_dir, f'Test_225_unconstr_tanh_test_{test_idx}')
        
        # Cerca il file csv per il test corrente
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
        if len(test_files) == 0:
            print(f"No CSV file found for test {test_idx}")
            continue

        # Supponiamo che ci sia un solo file CSV per test
        csv_file = os.path.join(test_dir, test_files[0])
        
        # Leggi il file CSV come dataframe
        df = pd.read_csv(csv_file)

        # Supponiamo che le colonne del CSV siano 'Positions', 'Velocities'
        positions = df['Positions'].values
        velocities = df['Velocities'].values
        inputs = df['Inputs'].values
        terminal_cost = df['Terminal_Costs'].values
        total_cost = df['Total_Costs'].values

        # Aggiunge la traiettoria alle liste
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_inputs.append(inputs)
        all_terminal_costs.append(terminal_cost)
        all_total_costs.append(total_cost)

    # Crea i grafici con le traiettorie combinate
    plt.figure(figsize=(10, 8))

    # Plot delle posizioni (sottografico 1)
    plt.subplot(3, 1, 1)
    for idx, positions in enumerate(all_positions):
        plt.plot(positions)
    plt.xlabel('MPC Step')
    plt.ylabel('q [rad]')
    plt.title('Positions Over All Tests')
    plt.grid(True)

    # Plot delle velocità (sottografico 2)
    plt.subplot(3, 1, 2)
    for idx, velocities in enumerate(all_velocities):
        plt.plot(velocities)
    plt.xlabel('MPC Step')
    plt.ylabel('v [rad/s]')
    plt.title('Velocities Over All Tests')
    plt.grid(True)

    # Plot delle torque (sottografico 3)
    plt.subplot(3, 1, 3)
    for idx, inputs in enumerate(all_inputs):
        plt.plot(inputs)
    plt.xlabel('MPC Step')
    plt.ylabel('u [N/m]')
    plt.title('Inputs Over All Tests')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Now plot the terminal and total costs
    plt.figure(figsize=(10, 6))

    # Plot terminal costs
    plt.subplot(2, 1, 1)
    for idx, terminal_cost in enumerate(all_terminal_costs):
        plt.plot(terminal_cost, label=f'Test {idx+1}')
    plt.xlabel('MPC Step')
    plt.ylabel('Terminal Cost')
    plt.title('Terminal Cost Over All Tests')
    plt.grid(True)
    
    # Plot total costs
    plt.subplot(2, 1, 2)
    for idx, total_cost in enumerate(all_total_costs):
        plt.plot(total_cost, label=f'Test {idx+1}')
    plt.xlabel('MPC Step')
    plt.ylabel('Total Cost')
    plt.title('Total Cost Over All Tests')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Esempio di utilizzo:
output_dir = 'Plots_&_Animations'  # Cartella dove sono salvati i risultati
num_tests = 50  # Numero totale di test eseguiti
plot_all_trajectories(output_dir, num_tests)
