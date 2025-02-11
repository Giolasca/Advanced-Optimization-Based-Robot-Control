import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the number of tests
num_tests = 15  
noise = 1

# Iterate over each test number
for test_number in range(1, num_tests + 1):
    # Load CSV files for each test
    if noise:
        tc_path = f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_TC_noise_225_unconstr_tanh_test_{test_number}.csv'
        ntc_path = f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_NTC_noise_225_unconstr_tanh_test_{test_number}.csv'
    else:
        tc_path = f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_TC_225_unconstr_tanh_test_{test_number}.csv'
        ntc_path = f'Plots_&_Animations/Test_225_unconstr_tanh_test_{test_number}/mpc_SP_NTC_225_unconstr_tanh_test_{test_number}.csv'

    # Read the CSV files into DataFrames
    TC = pd.read_csv(tc_path)
    NTC = pd.read_csv(ntc_path)

    # Parameters 
    N_TC = 1
    N_NTC = 100
    q_target = 5/4*np.pi

    # Extract columns from each DataFrame for TC
    pos_TC, vel_TC, Input_TC = TC['Positions'].to_numpy(), TC['Velocities'].to_numpy(), TC['Inputs'].to_numpy()
    Total_Cost_TC, Terminal_Cost_TC = TC['Total_Costs'].to_numpy(), TC['Terminal_Costs'].to_numpy()

    # Extract columns from each DataFrame for NTC
    pos_NTC, vel_NTC, Input_NTC = NTC['Positions'].to_numpy(), NTC['Velocities'].to_numpy(), NTC['Inputs'].to_numpy()
    Total_Cost_NTC = NTC['Total_Costs'].to_numpy()

    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Subplot 1: Position over MPC steps
    axs[0].plot(pos_TC, marker='o', linestyle='', label=f'Position (TC) - N = {N_TC}', color='b')  
    axs[0].plot(pos_NTC, marker='', linestyle='-', label=f'Position (NTC) - N = {N_NTC}', color='r')  
    axs[0].axhline(y=(q_target), color='g', linestyle='--', label=f'Target (y={q_target})')  
    axs[0].set_xlabel('MPC Step')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Comparative Plot of Position over MPC Steps')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Velocity over MPC steps
    axs[1].plot(vel_TC, marker='o', linestyle='', label=f'Velocity (TC) - N = {N_TC}', color='b')  
    axs[1].plot(vel_NTC, marker='', linestyle='-', label=f'Velocity (NTC) - N = {N_NTC}', color='r')  
    axs[1].axhline(y=(0), color='g', linestyle='--', label=f'Target (y={0})')  
    axs[1].set_xlabel('MPC Step')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Comparative Plot of Velocity over MPC Steps')
    axs[1].legend()
    axs[1].grid(True)

    # Subplot 3: Control input over MPC steps
    axs[2].plot(Input_TC, marker='o', linestyle='', label=f'COntrol Input (TC) - N = {N_TC}', color='b')  
    axs[2].plot(Input_NTC, marker='', linestyle='-', label=f'COntrol Input (NTC) - N = {N_NTC}', color='r')  
    axs[2].set_xlabel('MPC Step')
    axs[2].set_ylabel('Control Input')
    axs[2].set_title('Comparative Plot of Control Input over MPC Steps')
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

    # Comparative plot of Total Cost over time
    plt.figure(figsize=(10, 6))
    plt.plot(Total_Cost_TC, label='Total Cost TC', color='b')
    plt.plot(Total_Cost_NTC, label='Total Cost NTC', color='r')
    plt.legend()
    plt.xlabel('mpc_step')
    plt.ylabel('Cost')
    plt.title('Comparative Plot of Total Cost over Time')
    plt.grid(True)
    plt.show()