import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = 225

# Load data from CSV files
if(test == 135):
    TC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC_135_unconstr_tanh.csv')
    NTC = pd.read_csv('Plots_&_Animations/mpc_SP_TC_135_unconstr_tanh.csv')
    q_target = 3/4 * np.pi
    N_TC = 30
    N_NTC = 1
if(test == 180):
    TC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC_180_unconstr_tanh.csv')
    NTC = pd.read_csv('Plots_&_Animations/mpc_SP_TC_180_unconstr_tanh.csv')
    q_target = 4/4 * np.pi
    N_TC = 30
    N_NTC = 1
if(test == 225):
    TC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC_225_unconstr_tanh.csv')
    NTC = pd.read_csv('Plots_&_Animations/mpc_SP_TC_225_unconstr_tanh.csv')
    q_target = 5/4 * np.pi
    N_TC = 30
    N_NTC = 1

# Extract columns from each DataFrame for TC
pos_TC, vel_TC, Input_TC = TC['Positions'].to_numpy(), TC['Velocities'].to_numpy(), TC['Inputs'].to_numpy()
Total_Cost_TC, Terminal_Cost_TC = TC['Total_Costs'].to_numpy(), TC['Terminal_Costs'].to_numpy()

# Extract columns from each DataFrame for NTC
pos_NTC, vel_NTC, Input_NTC = NTC['Positions'].to_numpy(), NTC['Velocities'].to_numpy(), NTC['Inputs'].to_numpy()
Total_Cost_NTC, Terminal_Cost_NTC = NTC['Total_Costs'].to_numpy(), NTC['Terminal_Costs'].to_numpy()

# Comparative plot of Position vs Velocity with markers
plt.figure(figsize=(10, 6))
plt.plot(pos_TC, vel_TC, marker='o', linestyle='-', label=f'Time_horizon = {N_TC} (TC)')
plt.plot(pos_NTC, vel_NTC, marker='o', linestyle='-', label=f'Time_horizon = {N_NTC} (NTC)')
plt.legend()
plt.xlabel('Velocity')
plt.ylabel('Position')
plt.title('Comparative Plot of Position vs Velocity for different N values')
plt.grid(True)
plt.show()

# Comparative plot of Input over time
plt.figure(figsize=(10, 6))
plt.plot(Input_TC, label=f'Time_horizon = {N_TC} (TC)')
plt.plot(Input_NTC, label=f'Time_horizon = {N_NTC} (NTC)')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Input')
plt.title('Comparative Plot of Input over Time')
plt.grid(True)
plt.show()

# Comparative plot of Total Cost over time
plt.figure(figsize=(10, 6))
plt.plot(Total_Cost_TC, label='Total Cost TC')
plt.plot(Total_Cost_NTC, label='Total Cost NTC')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Cost')
plt.title('Comparative Plot of Total Cost over Time')
plt.grid(True)
plt.show()

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

# Create figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Subplot 1: Position over MPC steps
axs[0].plot(pos_TC, marker='', linestyle='-', label=f'Position (TC) - N = {N_TC}', color='b')  
axs[0].plot(pos_NTC, marker='', linestyle='-', label=f'Position (NTC) - N = {N_NTC}', color='r')  
axs[0].axhline(y=(q_target), color='g', linestyle='--', label=f'Threshold (y={q_target})')  
axs[0].set_xlabel('MPC Step')
axs[0].set_ylabel('Position')
axs[0].set_title('Comparative Plot of Position over MPC Steps')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Velocity over MPC steps
axs[1].plot(vel_TC, marker='', linestyle='-', label=f'Velocity (TC) - N = {N_TC}', color='b')  
axs[1].plot(vel_NTC, marker='', linestyle='-', label=f'Velocity (NTC) - N = {N_NTC}', color='r')  
axs[1].axhline(y=(0), color='g', linestyle='--', label=f'Target (y={0})') 
axs[1].set_xlabel('MPC Step')
axs[1].set_ylabel('Velocity')
axs[1].set_title('Comparative Plot of Velocity over MPC Steps')
axs[1].legend()
axs[1].grid(True)

# Subplot 3: Control input over MPC steps
axs[2].plot(Input_TC, marker='', linestyle='-', label=f'COntrol Input (TC) - N = {N_TC}', color='b')  
axs[2].plot(Input_NTC, marker='', linestyle='-', label=f'COntrol Input (NTC) - N = {N_NTC}', color='r')  
axs[2].set_xlabel('MPC Step')
axs[2].set_ylabel('Control Input')
axs[2].set_title('Comparative Plot of Control Input over MPC Steps')
axs[2].legend()
axs[2].grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()