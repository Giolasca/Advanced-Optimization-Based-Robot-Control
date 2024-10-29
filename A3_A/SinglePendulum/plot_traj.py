import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_system_trajectories(tc_files, ntc_files, target_position, target_velocity):
    """
    Plot system trajectories for TC (Test Cases) and NTC (No Test Cases) with initial states and a target state.

    Parameters:
    - tc_files (list): List of file paths for the TC CSV files.
    - ntc_files (list): List of file paths for the NTC CSV files.
    - target_position (float): Target position in radians.
    - target_velocity (float): Target velocity in m/s.
    """
    # Initialize the plot
    plt.figure(figsize=(10, 8))
    plt.title("System Trajectories")
    plt.xlabel("Position [rad]")
    plt.ylabel("Velocity [m/s]")

    # Color map and line styles for TC and NTC
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'black']
    linestyles = ['-', '--']  # Solid for TC, dashed for NTC

    # Plot the test trajectories for TC and NTC
    for i in range(len(tc_files)):
        tc_file = tc_files[i]
        ntc_file = ntc_files[i]
        color = colors[i % len(colors)]
        
        # Plot TC trajectory
        if os.path.exists(tc_file):
            data = pd.read_csv(tc_file)
            positions = data['Positions'].to_numpy()
            velocities = data['Velocities'].to_numpy()

            plt.plot(positions, velocities, color=color, linestyle=linestyles[0],
                     label=f'TC Test {i+1}' if i == 0 else "")
            
            # Mark the initial state for TC
            plt.plot(positions[0], velocities[0], marker='o', markersize=8,
                     markeredgecolor='black', markerfacecolor='yellow')
            plt.text(positions[0], velocities[0],
                     f'Sim-{i+1}\n $x_0 = [{positions[0]:.2f}; {velocities[0]:.2f}]$', 
                     fontsize=10, color='black')

        # Plot NTC trajectory
        if os.path.exists(ntc_file):
            data = pd.read_csv(ntc_file)
            positions = data['Positions'].to_numpy()
            velocities = data['Velocities'].to_numpy()

            plt.plot(positions, velocities, color=color, linestyle=linestyles[1],
                     label=f'NTC Test {i+1}' if i == 0 else "")

    # Highlight the target state with a distinct marker
    plt.plot(target_position, target_velocity, marker='X', markersize=10,
             markeredgecolor='black', markerfacecolor='red', label='Target State')

    # Display legend and grid
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# Usage example
# Define the number of tests
num_tests = 15  

# Generate file paths for TC and NTC files
tc_files = [
    f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_TC_225_unconstr_tanh_test_{test_number}.csv'
    for test_number in range(1, num_tests + 1)
]

ntc_files = [
    f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_NTC_225_unconstr_tanh_test_{test_number}.csv'
    for test_number in range(1, num_tests + 1)
]

# Ora puoi usare tc_files e ntc_files come parametri per la funzione di plotting
# Ad esempio:
target_position = 5/4 * np.pi  # Target position in radians
target_velocity = 0  # Target velocity in m/s

plot_system_trajectories(tc_files, ntc_files, target_position, target_velocity)
