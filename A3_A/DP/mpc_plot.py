import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
#TC = pd.read_csv('Plots_&_Animations/MPC_SinglePendulum_NTC.csv')
NTC = pd.read_csv('Plots_&_Animations/MPC_DoublePendulum_NTC.csv')

# Extract columns from each DataFrame for TC
#pos_q1_TC, vel_q1_TC, Input_u1_TC = TC['q1'].to_numpy(), TC['v1'].to_numpy(), TC['u1'].to_numpy()
#pos_q2_TC, vel_q2_TC, Input_u2_TC = TC['q2'].to_numpy(), TC['v2'].to_numpy(), TC['u2'].to_numpy()
#Total_Cost_TC, Terminal_Cost_TC = TC['Total_Costs'].to_numpy(), TC['Terminal_Costs'].to_numpy()

# Extract columns from each DataFrame for TC
pos_q1_NTC, vel_q1_NTC, Input_u1_NTC = NTC['q1'].to_numpy(), NTC['v1'].to_numpy(), NTC['u1'].to_numpy()
pos_q2_NTC, vel_q2_NTC, Input_u2_NTC = NTC['q2'].to_numpy(), NTC['v2'].to_numpy(), NTC['u2'].to_numpy()
Total_Cost_NTC, Terminal_Cost_NTC = NTC['Total_Costs'].to_numpy(), NTC['Terminal_Costs'].to_numpy()

# Comparative plot of Position vs Velocity with markers
plt.figure(figsize=(10, 6))
#plt.plot(pos_q1_TC, vel_q1_TC, marker='o', linestyle='-', label='First link - Time_horizon = 0.4 (TC)')
#plt.plot(pos_q2_TC, vel_q2_TC, marker='o', linestyle='-', label='Second link - Time_horizon = 0.4 (TC)')
plt.plot(pos_q1_NTC, vel_q1_NTC, marker='o', linestyle='-', label='Pendulum 1 - Time_horizon = 0.8 (NTC)')
plt.plot(pos_q2_NTC, vel_q2_NTC, marker='o', linestyle='-', label='Pendulum 2 - Time_horizon = 0.8 (NTC)')
plt.legend()
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Comparative Plot of Position vs Velocity for different N values')
plt.grid(True)
plt.show()

# Comparative plot of Position over time
plt.figure(figsize=(10, 6))
#plt.plot(Input_u1_TC, label='Time_horizon = 0.4 (TC)')
#plt.plot(Input_u2_TC, label='Time_horizon = 0.4 (TC)')
plt.plot(pos_q1_NTC, label='Pendulum 1 - Time_horizon = 0.8 (NTC)')
plt.plot(pos_q2_NTC, label='Pendulum 2 - Time_horizon = 0.8 (NTC)')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Positions')
plt.title('Comparative Plot of Positions over Time')
plt.grid(True)
plt.show()


# Comparative plot of Input over time
plt.figure(figsize=(10, 6))
#plt.plot(Input_u1_TC, label='Time_horizon = 0.4 (TC)')
#plt.plot(Input_u2_TC, label='Time_horizon = 0.4 (TC)')
plt.plot(Input_u1_NTC, label='Pendulum 1 - Time_horizon = 0.8 (NTC)')
plt.plot(Input_u2_NTC, label='Pendulum 2 - Time_horizon = 0.8 (NTC)')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Input')
plt.title('Comparative Plot of Input over Time')
plt.grid(True)
plt.show()

# Comparative plot of Total Cost over time
plt.figure(figsize=(10, 6))
#plt.plot(Total_Cost_TC, label='Total Cost TC')
plt.plot(Total_Cost_NTC, label='Total Cost NTC')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Cost')
plt.title('Comparative Plot of Total Cost over Time')
plt.grid(True)
plt.show()

# Comparative plot of Terminal Cost over time
plt.figure(figsize=(10, 6))
#plt.plot(Terminal_Cost_NTC, label='Terminal Cost TC')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Input')
plt.title('Comparative Plot of Terminal COst over Time')
plt.grid(True)
plt.show()



TEST = pd.read_csv('Plots_&_Animations/DoublePendulum.csv')

# Extract columns from each DataFrame for TC
pos_q1_TEST = TEST['Positions_q1'].to_numpy()
pos_q2_TEST = TEST['Positions_q2'].to_numpy()

# Comparative plot of Position over time
plt.figure(figsize=(10, 6))
#plt.plot(Input_u1_TC, label='Time_horizon = 0.4 (TC)')
#plt.plot(Input_u2_TC, label='Time_horizon = 0.4 (TC)')
plt.plot(pos_q1_TEST, label='Pendulum 1 - Time_horizon = 0.8 (TEST)')
plt.plot(pos_q2_TEST, label='Pendulum 2 - Time_horizon = 0.8 (TEST)')
plt.legend()
plt.xlabel('mpc_step')
plt.ylabel('Positions')
plt.title('Comparative Plot of Positions over Time')
plt.grid(True)
plt.show()