import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
TC = pd.read_csv('Plots_&_Animations/mpc_SP_TC.csv')
NTC = pd.read_csv('Plots_&_Animations/mpc_SP_NTC.csv')

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
plt.plot(Input_TC, label='Time_horizon = 0.4 (TC)')
plt.plot(Input_NTC, label='Time_horizon = 0.8 (NTC)')
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