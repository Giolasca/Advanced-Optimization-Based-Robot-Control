import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = 180

# Load data from CSV files
if(test == 135):
    TC = pd.read_csv('Plots_&_Animations/mpc_SP_TC_135_v1.csv')
    NTC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC_135_v1.csv')
if(test == 180):
    TC = pd.read_csv('Plots_&_Animations/mpc_SP_TC_180_v1.csv')
    NTC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC_180_v1.csv')
if(test == 225):
    TC = pd.read_csv('Plots_&_Animations/mpc_SP_TC_225_v1.csv')
    NTC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC_225_v1.csv')

# Extract columns from each DataFrame for TC
pos_TC, vel_TC, Input_TC = TC['Positions'].to_numpy(), TC['Velocities'].to_numpy(), TC['Inputs'].to_numpy()
Total_Cost_TC, Terminal_Cost_TC = TC['Total_Costs'].to_numpy(), TC['Terminal_Costs'].to_numpy()

# Extract columns from each DataFrame for NTC
pos_NTC, vel_NTC, Input_NTC = NTC['Positions'].to_numpy(), NTC['Velocities'].to_numpy(), NTC['Inputs'].to_numpy()
Total_Cost_NTC, Terminal_Cost_NTC = NTC['Total_Costs'].to_numpy(), NTC['Terminal_Costs'].to_numpy()

# Comparative plot of Position vs Velocity with markers
plt.figure(figsize=(10, 6))
plt.plot(pos_TC, vel_TC, marker='o', linestyle='-', label='Time_horizon = 0.4 (TC)')
plt.plot(pos_NTC, vel_NTC, marker='o', linestyle='-', label='Time_horizon = 0.8 (NTC)')
plt.legend()
plt.xlabel('Velocity')
plt.ylabel('Position')
plt.title('Comparative Plot of Position vs Velocity for different N values')
plt.grid(True)
plt.show()

# Comparative plot of Input over time
plt.figure(figsize=(10, 6))
plt.plot(Input_TC, label='Time_horizon = 1 (TC)')
plt.plot(Input_NTC, label='Time_horizon = 30 (NTC)')
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

# Comparative plot of Terminal Cost over time
plt.figure(figsize=(10, 6))
plt.plot(Terminal_Cost_TC, label='Terminal Cost TC')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Input')
plt.title('Comparative Plot of Terminal Cost over Time')
plt.grid(True)
plt.show()

# Create figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Subplot 1: Position over MPC steps
axs[0].plot(pos_TC, marker='o', linestyle='', label='Position (TC) - N = 4', color='b')  # Dati TC con puntini
axs[0].plot(pos_NTC, marker='', linestyle='-', label='Position (NTC) - N = 30', color='r')  # Dati NTC
axs[0].axhline(y=(test*np.pi/180), color='g', linestyle='--', label=f'Threshold (y={test*np.pi/180})')  # Linea orizzontale tratteggiata verde
axs[0].set_xlabel('MPC Step')
axs[0].set_ylabel('Position')
axs[0].set_title('Comparative Plot of Position over MPC Steps')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Velocity over MPC steps
axs[1].plot(vel_TC, marker='o', linestyle='', label='Velocity (TC) - N = 4', color='b')  # Dati TC con puntini
axs[1].plot(vel_NTC, marker='', linestyle='-', label='Velocity (NTC) - N = 30', color='r')  # Dati NTC
axs[1].set_xlabel('MPC Step')
axs[1].set_ylabel('Velocity')
axs[1].set_title('Comparative Plot of Velocity over MPC Steps')
axs[1].legend()
axs[1].grid(True)

# Subplot 3: Control input over MPC steps
axs[2].plot(Input_TC, marker='o', linestyle='', label='Control Input (TC) - N = 4', color='b')  # Dati TC con puntini
axs[2].plot(Input_NTC, marker='', linestyle='-', label='Control Input (NTC) - N = 30', color='r')  # Dati NTC
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
axs[0].plot(pos_TC, marker='', linestyle='-', label='Position (TC) - N = 4', color='b')  # Dati TC con puntini
axs[0].plot(pos_NTC, marker='', linestyle='-', label='Position (NTC) - N = 30', color='r')  # Dati NTC
axs[0].axhline(y=(test*np.pi/180), color='g', linestyle='--', label=f'Threshold (y={test*np.pi/180})')  # Linea orizzontale tratteggiata verde
axs[0].set_xlabel('MPC Step')
axs[0].set_ylabel('Position')
axs[0].set_title('Comparative Plot of Position over MPC Steps')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Velocity over MPC steps
axs[1].plot(vel_TC, marker='', linestyle='-', label='Velocity (TC) - N = 4', color='b')  # Dati TC con puntini
axs[1].plot(vel_NTC, marker='', linestyle='-', label='Velocity (NTC) - N = 30', color='r')  # Dati NTC
axs[1].set_xlabel('MPC Step')
axs[1].set_ylabel('Velocity')
axs[1].set_title('Comparative Plot of Velocity over MPC Steps')
axs[1].legend()
axs[1].grid(True)

# Subplot 3: Control input over MPC steps
axs[2].plot(Input_TC, marker='', linestyle='-', label='Control Input (TC) - N = 4', color='b')  # Dati TC con puntini
axs[2].plot(Input_NTC, marker='', linestyle='-', label='Control Input (NTC) - N = 30', color='r')  # Dati NTC
axs[2].set_xlabel('MPC Step')
axs[2].set_ylabel('Control Input')
axs[2].set_title('Comparative Plot of Control Input over MPC Steps')
axs[2].legend()
axs[2].grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()