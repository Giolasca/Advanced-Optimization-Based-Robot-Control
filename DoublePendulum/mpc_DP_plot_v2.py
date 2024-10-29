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
    v_ref = 0

    # Extract columns for TC and NTC
    q1_TC, v1_TC, q2_TC, v2_TC, u1_TC, u2_TC = TC['q1'], TC['v1'], TC['q2'], TC['v2'], TC['u1'], TC['u2']
    q1_NTC, v1_NTC, q2_NTC, v2_NTC, u1_NTC, u2_NTC = NTC['q1'], NTC['v1'], NTC['q2'], NTC['v2'], NTC['u1'], NTC['u2']

    # Create figure with subplots 3x1
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))

    # First subplot (q1 and q2)
    axs[0].plot(q1_NTC, label='Position_q1 (NTC) - N={}'.format(N_NTC), color='red', linewidth=2)  # NTC - q1
    axs[0].plot(q1_TC, label='Position_q1 (TC) - N={}'.format(N_TC), color='blue', linestyle='--', linewidth=2)  # TC - q1
    axs[0].plot(q2_NTC, label='Position_q2 (NTC) - N={}'.format(N_NTC), color='orange', linewidth=2)  # NTC - q2
    axs[0].plot(q2_TC, label='Position_q2 (TC) - N={}'.format(N_TC), color='cyan', linestyle='--', linewidth=2)  # TC - q2
    axs[0].axhline(y=q1_target, color='g', linestyle='--', label='q_ref = π', linewidth=2)
    axs[0].set_xlabel('MPC Step')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Comparative Plot of q1 and q2')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot (v1 and v2)
    axs[1].plot(v1_NTC, label='Velocity_v1 (NTC) - N={}'.format(N_NTC), color='red', linewidth=2)  # NTC - v1
    axs[1].plot(v1_TC, label='Velocity_v1 (TC) - N={}'.format(N_TC), color='blue', linestyle='--', linewidth=2)  # TC - v1
    axs[1].plot(v2_NTC, label='Velocity_v2 (NTC) - N={}'.format(N_NTC), color='orange', linewidth=2)  # NTC - v2
    axs[1].plot(v2_TC, label='Velocity_v2 (TC) - N={}'.format(N_TC), color='cyan', linestyle='--', linewidth=2)  # TC - v2
    axs[1].axhline(y=v_ref, color='g', linestyle='--', label='v_ref = 0', linewidth=2)
    axs[1].set_xlabel('MPC Step')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Comparative Plot of v1 and v2')
    axs[1].legend()
    axs[1].grid(True)

    # Third subplot (u1 and u2)
    axs[2].plot(u1_NTC, label='Input_u1 (NTC) - N={}'.format(N_NTC), color='red', linewidth=2)  # NTC - u1
    axs[2].plot(u1_TC, label='Input_u1 (TC) - N={}'.format(N_TC), color='blue', linestyle='--', linewidth=2)  # TC - u1
    axs[2].plot(u2_NTC, label='Input_u2 (NTC) - N={}'.format(N_NTC), color='orange', linewidth=2)  # NTC - u2
    axs[2].plot(u2_TC, label='Input_u2 (TC) - N={}'.format(N_TC), color='cyan', linestyle='--', linewidth=2)  # TC - u2
    axs[2].set_xlabel('MPC Step')
    axs[2].set_ylabel('Control Input')
    axs[2].set_title('Comparative Plot of u1 and u2')
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the plot for the current test
    plt.show()