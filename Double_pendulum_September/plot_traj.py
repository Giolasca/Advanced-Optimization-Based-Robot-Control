import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read data from a CSV file
def read_csv_data(file_path):
    return pd.read_csv(file_path)

# Function to plot the data in subplots
def plot_trajectories(file1, file2):
    # Read the CSV files
    data1 = read_csv_data(file1)
    data2 = read_csv_data(file2)
    
    # Create the figure and subplots with a 3x2 layout
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # First subplot (q1_file1 and q1_file2)
    q1_ref = np.pi
    axs[0, 0].plot(data1['q1'], label='q1 with TC: T = 0.01 & N = 5', color='b')
    axs[0, 0].plot(data2['q1'], label='q1 without TC: T = 1 & N = 100', color='r')
    axs[0, 0].axhline(y=q1_ref, color='g', linestyle='--', label='q1_ref = π')
    axs[0, 0].set_title('q1 trajectory comparison')
    axs[0, 0].set_xlabel('Time step')
    axs[0, 0].set_ylabel('q1 [rad]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Second subplot (q2_file1 and q2_file2)
    q2_ref = np.pi
    axs[0, 1].plot(data1['q2'], label='q2 with TC: T = 0.01 & N = 5', color='b')
    axs[0, 1].plot(data2['q2'], label='q2 without TC: T = 1 & N = 100', color='r')
    axs[0, 1].axhline(y=q2_ref, color='g', linestyle='--', label='q2_ref = π')
    axs[0, 1].set_title('q2 trajectory comparison')
    axs[0, 1].set_xlabel('Time step')
    axs[0, 1].set_ylabel('q2 [rad]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Third subplot (v1_file1 and v1_file2)
    v_ref = 0
    axs[1, 0].plot(data1['v1'], label='v1 with TC: T = 0.01 & N = 5', color='b')
    axs[1, 0].plot(data2['v1'], label='v1 without TC: T = 1 & N = 100', color='r')
    axs[1, 0].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0')
    axs[1, 0].set_title('v1 trajectory comparison')
    axs[1, 0].set_xlabel('Time step')
    axs[1, 0].set_ylabel('v1 [rad/s]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Fourth subplot (v2_file1 and v2_file2)
    axs[1, 1].plot(data1['v2'], label='v2 with TC: T = 0.01 & N = 5', color='b')
    axs[1, 1].plot(data2['v2'], label='v2 without TC: T = 1 & N = 100', color='r')
    axs[1, 1].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0')
    axs[1, 1].set_title('v2 trajectory comparison')
    axs[1, 1].set_xlabel('Time step')
    axs[1, 1].set_ylabel('v2 [rad/s]')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Fifth subplot (u1_file1 and u1_file2)
    axs[2, 0].plot(data1['u1'], label='u1 with TC: T = 0.01 & N = 5', color='b')
    axs[2, 0].plot(data2['u1'], label='u1 without TC: T = 1 & N = 100', color='r')
    axs[2, 0].set_title('u1 trajectory comparison')
    axs[2, 0].set_xlabel('Time step')
    axs[2, 0].set_ylabel('u1 [N/m]')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # Sixth subplot (u2_file1 and u2_file2)
    axs[2, 1].plot(data1['u2'], label='u2 with TC: T = 0.01 & N = 5', color='b')
    axs[2, 1].plot(data2['u2'], label='u2 without TC: T = 1 & N = 100', color='r')
    axs[2, 1].set_title('u2 trajectory comparison')
    axs[2, 1].set_xlabel('Time step')
    axs[2, 1].set_ylabel('u2 [N/m]')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    # Add a clean layout
    plt.tight_layout()
    plt.show()

    # Separate plot for total cost - terminal cost
    if 'Total_Costs' in data1.columns and 'Terminal_Costs' in data1.columns:
        total_cost1_adjusted = data1['Total_Costs'] - data1['Terminal_Costs']
    else:
        print("Columns 'Total_Costs' or 'Terminal_Cost' missing in file1")
        return

    if 'Total_Costs' in data2.columns:
        total_cost2 = data2['Total_Costs']
    else:
        print("Column 'Total_Costs' missing in file2")
        return
    
    # Plot for total cost
    plt.figure(figsize=(10, 6))
    plt.plot(total_cost1_adjusted, label='Total Cost (Adjusted) with TC: T = 0.01 & N = 5', color='b')
    plt.plot(total_cost2, label='Total Cost without TC: T = 1 & N = 100', color='r')
    
    plt.title('Total Cost Comparison (Adjusted for File 1)')
    plt.xlabel('Time step')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

# Specify the paths to the CSV files
file1 = 'Plots_&_Animations/MPC_DoublePendulum_TC.csv'
file2 = 'Plots_&_Animations/MPC_DoublePendulum_NTC_T_1.csv'

# Execute the plotting
plot_trajectories(file1, file2)
