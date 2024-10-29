import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the number of tests
num_tests = 15  

# Iterate over each test number
for test_number in range(1, num_tests + 1):
    # Load CSV files for each test
    tc_path = f'Plots_&_Animations/Test_180_180_unconstr_test_{test_number}/mpc_DP_TC_180_180_unconstr_test_{test_number}.csv'
    ntc_path = f'Plots_&_Animations/Test_180_180_unconstr_test_{test_number}/mpc_DP_NTC_180_180_unconstr_test_{test_number}.csv'
    
    # Read the CSV files into DataFrames
    TC = pd.read_csv(tc_path)
    NTC = pd.read_csv(ntc_path)
    
    # Parameters 
    N_TC = 5
    N_NTC = 100
    q1_target = np.pi

    # Extract columns for TC and NTC
    q1_TC, v1_TC, q2_TC, v2_TC, u1_TC, u2_TC = TC['q1'], TC['v1'], TC['q2'], TC['v2'], TC['u1'], TC['u2']
    q1_NTC, v1_NTC, q2_NTC, v2_NTC, u1_NTC, u2_NTC = NTC['q1'], NTC['v1'], NTC['q2'], NTC['v2'], NTC['u1'], NTC['u2']

    # Create figure with subplots 3x2
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Aggiungere un titolo generale
    fig.suptitle(f'Comparative Analysis of TC and NTC for Test {test_number}', fontsize=16)

    # First subplot (q1)
    axs[0, 0].plot(q1_TC, label=f'Position_q1 (TC) - N = {N_TC}', color='b')
    axs[0, 0].plot(q1_NTC, label=f'Position_q1 (NTC) - N = {N_NTC}', color='r')
    axs[0, 0].axhline(y=q1_target, color='g', linestyle='--', label='q1_ref = π')
    axs[0, 0].set_xlabel('MPC Step')
    axs[0, 0].set_ylabel('q1')
    axs[0, 0].set_title('Comparative Plot of q1')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Second subplot (q2)
    axs[0, 1].plot(q2_TC, label=f'Position_q2 (TC) - N={N_TC}', color='b')
    axs[0, 1].plot(q2_NTC, label=f'Position_q2 (NTC) - N={N_NTC}', color='r')
    axs[0, 1].axhline(y=q1_target, color='g', linestyle='--', label='q2_ref = π')
    axs[0, 1].set_xlabel('MPC Step')
    axs[0, 1].set_ylabel('q2')
    axs[0, 1].set_title('Comparative Plot of q2')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Third subplot (v1)
    v_ref = 0
    axs[1, 0].plot(v1_TC, label=f'Velocity_v1 (TC) - N={N_TC}', color='b')
    axs[1, 0].plot(v1_NTC, label=f'Velocity_v1 (NTC) - N={N_NTC}', color='r')
    axs[1, 0].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0')
    axs[1, 0].set_xlabel('MPC Step')
    axs[1, 0].set_ylabel('v1')
    axs[1, 0].set_title('Comparative Plot of v1')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Fourth subplot (v2)
    axs[1, 1].plot(v2_TC, label=f'Velocity_v2 (TC) - N={N_TC}', color='b')
    axs[1, 1].plot(v2_NTC, label=f'Velocity_v2 (NTC) - N={N_NTC}', color='r')
    axs[1, 1].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0')
    axs[1, 1].set_xlabel('MPC Step')
    axs[1, 1].set_ylabel('v2')
    axs[1, 1].set_title('Comparative Plot of v2')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Fifth subplot (u1)
    axs[2, 0].plot(u1_TC, label=f'Input_u1 (TC) - N={N_TC}', color='b')
    axs[2, 0].plot(u1_NTC, label=f'Input_u1 (NTC) - N={N_NTC}', color='r')
    axs[2, 0].set_xlabel('MPC Step')
    axs[2, 0].set_ylabel('u1')
    axs[2, 0].set_title('Comparative Plot of u1')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Sixth subplot (u2)
    axs[2, 1].plot(u2_TC, label=f'Input_u2 (TC) - N={N_TC}', color='b')
    axs[2, 1].plot(u2_NTC, label=f'Input_u2 (NTC) - N={N_NTC}', color='r')
    axs[2, 1].set_xlabel('MPC Step')
    axs[2, 1].set_ylabel('u2')
    axs[2, 1].set_title('Comparative Plot of u2')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the plot for the current test
    plt.show()

