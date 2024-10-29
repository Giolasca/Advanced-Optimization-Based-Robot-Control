import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_trajectories(output_dir, num_tests):
    """
    Reads trajectories from each test and plots them on a single graph.
    
    :param output_dir: Directory where the results of each test are saved.
    :param num_tests: Total number of tests executed.
    """
    
    # Initialize lists to collect the trajectories
    all_positions = []
    all_velocities = []
    all_inputs = []
    
    # Loop through the generated files for each test
    for test_idx in range(1, num_tests + 1):
        test_dir = os.path.join(output_dir, f'Test_225_unconstr_tanh_test_{test_idx}')
        
        # Search for the CSV file for the current test
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
        if len(test_files) == 0:
            print(f"No CSV file found for test {test_idx}")
            continue  # Skip to the next test if no CSV file is found

        # Assume there is only one CSV file per test
        csv_file = os.path.join(test_dir, test_files[0])
        
        # Read the CSV file as a dataframe
        df = pd.read_csv(csv_file)

        # Assume the columns of the CSV are 'Positions', 'Velocities', and 'Inputs'
        positions = df['Positions'].values
        velocities = df['Velocities'].values
        inputs = df['Inputs'].values

        # Append the trajectory to the lists
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_inputs.append(inputs)

    # Create plots with the combined trajectories
    plt.figure(figsize=(10, 8))

    # Plot of positions (subplot 1)
    plt.subplot(3, 1, 1)
    for idx, positions in enumerate(all_positions):
        plt.plot(positions)  
    plt.xlabel('MPC Step')  
    plt.ylabel('q [rad]')  
    plt.title('Positions Over All Tests')  
    plt.grid(True) 

    # Plot of velocities (subplot 2)
    plt.subplot(3, 1, 2)
    for idx, velocities in enumerate(all_velocities):
        plt.plot(velocities)  
    plt.xlabel('MPC Step')  
    plt.ylabel('v [rad/s]')  
    plt.title('Velocities Over All Tests')  
    plt.grid(True)  

    # Plot of inputs (subplot 3)
    plt.subplot(3, 1, 3)
    for idx, inputs in enumerate(all_inputs):
        plt.plot(inputs) 
    plt.xlabel('MPC Step') 
    plt.ylabel('u [N/m]')  
    plt.title('Inputs Over All Tests') 
    plt.grid(True)  

    plt.tight_layout()  # Adjust layout to prevent overlap
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_dir, 'Plot_all_traj.png'))
    plt.show()  # Display the plots


def impact_plot(output_dir, num_tests):
    """
    Plot system trajectories for TC (Test Cases) and NTC (No Test Cases) with initial states and a target state.

    Parameters:
    - param output_dir: Directory where the results of each test are saved.
    - param num_tests: Total number of tests executed.
    """

    # Generate file paths for TC and NTC files
    tc_files = [
        f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_TC_225_unconstr_tanh_test_{test_number}.csv'
        for test_number in range(1, num_tests + 1)
    ]

    ntc_files = [
        f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_NTC_225_unconstr_tanh_test_{test_number}.csv'
        for test_number in range(1, num_tests + 1)
    ]

    target_position = 5/4 * np.pi  # Target position in radians
    target_velocity = 0  # Target velocity in m/s

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
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_dir, 'Impact_plot.png'))
    plt.show()


# Example usage:
output_dir = 'Plots_&_Animations'  
num_tests = 15  # Total number of tests executed
plot_all_trajectories(output_dir, num_tests)  
impact_plot(output_dir, num_tests)
