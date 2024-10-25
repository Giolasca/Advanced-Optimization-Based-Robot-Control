import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# List of CSV files for the 15 tests
csv_files = [
    'Plots_&_Animations/Test_225_unconstr_tanh_test_1/mpc_SP_TC_225_unconstr_tanh_test_1.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_2/mpc_SP_TC_225_unconstr_tanh_test_2.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_3/mpc_SP_TC_225_unconstr_tanh_test_3.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_4/mpc_SP_TC_225_unconstr_tanh_test_4.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_5/mpc_SP_TC_225_unconstr_tanh_test_5.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_6/mpc_SP_TC_225_unconstr_tanh_test_6.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_7/mpc_SP_TC_225_unconstr_tanh_test_7.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_8/mpc_SP_TC_225_unconstr_tanh_test_8.csv'

]

# Add the folder containing the full MPC problem CSV files
full_mpc_files = [
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_1/mpc_SP_NTC_T_1_225_unconstr_tanh_test_1.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_2/mpc_SP_NTC_T_1_225_unconstr_tanh_test_2.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_3/mpc_SP_NTC_T_1_225_unconstr_tanh_test_3.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_4/mpc_SP_NTC_T_1_225_unconstr_tanh_test_4.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_5/mpc_SP_NTC_T_1_225_unconstr_tanh_test_5.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_6/mpc_SP_NTC_T_1_225_unconstr_tanh_test_6.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_7/mpc_SP_NTC_T_1_225_unconstr_tanh_test_7.csv',
    'Plots_&_Animations_NTC/Test_225_unconstr_tanh_test_8/mpc_SP_NTC_T_1_225_unconstr_tanh_test_8.csv'

]
# Define the target state (example values, replace with actual target values)
target_position = 5/4*np.pi  # Target position in radians
target_velocity = 0  # Target velocity in m/s

# Initialize the plot
plt.figure(figsize=(10, 8))
plt.title("System Trajectories")
plt.xlabel(r"Position [rad]")
plt.ylabel(r"Velocity $[m/s]$")

# Set a consistent color for the test trajectories
test_color = 'blue'

# Plot the test trajectories
for i, file in enumerate(csv_files):
    if not os.path.exists(file):
        print(f"File {file} not found.")
        continue

    # Load the data
    data = pd.read_csv(file)
    
    # Assuming columns named 'Positions' and 'Velocities' in each CSV file
    positions = data['Positions'].to_numpy()
    velocities = data['Velocities'].to_numpy()

    # Plot the trajectory for this test
    plt.plot(positions, velocities, color=test_color)

    # Mark the initial state with a distinct marker
    initial_velocity = velocities[0]
    initial_position = positions[0]

    # Mark the initial state with a distinct marker
    plt.plot(initial_position, initial_velocity, marker='o', markersize=8, 
             markeredgecolor='black', markerfacecolor='yellow')
    
    # Create the text annotation for the initial state with LaTeX formatting
    initial_state_text = f'Sim-{i+1}\n $x_0 = [{initial_position:.2f}; {initial_velocity:.2f}]$'
    plt.text(initial_position, initial_velocity, initial_state_text, fontsize=10, color='black')

# Add a single label for LTC-MPC trajectories
plt.plot([], [], color=test_color, label='LTC-MPC')  # Placeholder for legend

# Plot the full MPC trajectories
full_mpc_color = 'red'  # Choose a contrasting color (e.g., green)
for file in full_mpc_files:
    if not os.path.exists(file):
        print(f"File {file} not found.")
        continue

    # Load the data
    data = pd.read_csv(file)
    
    # Assuming columns named 'Positions' and 'Velocities' in each CSV file
    positions = data['Positions'].to_numpy()
    velocities = data['Velocities'].to_numpy()

    # Plot the full MPC trajectory with a solid line
    plt.plot(positions, velocities, color=full_mpc_color)

# Add a label for Full MPC trajectories
plt.plot([], [], color=full_mpc_color, label='Full MPC')  # Placeholder for legend

# Highlight the target state with a distinct marker
plt.plot(target_position, target_velocity, marker='X', markersize=10, 
         markeredgecolor='black', markerfacecolor='red', label='Target State')

# Display legend and grid
plt.legend(loc="best")
plt.grid(True)
plt.show()