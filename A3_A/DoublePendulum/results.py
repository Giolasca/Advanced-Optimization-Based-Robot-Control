import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files to read for the double pendulum
csv_files = [
    f'Plots_&_Animations/Test_180_180_unconstr_test_{test_number}/mpc_DP_TC_180_180_unconstr_test_{test_number}.csv'
    for test_number in range(1, 15)
]

# Initialize empty lists to store data from all files
terminal_costs = []
true_terminal_costs = []
initial_states = []
error = []  

# Loop through each CSV file and read the required columns
for file in csv_files:
    # Read CSV file
    df = pd.read_csv(file)

    # Assuming the columns are named 'Terminal_Costs' and 'True_Terminal_Costs'
    terminal_costs.append(df['Terminal_Costs'])
    true_terminal_costs.append(df['True_Terminal_Costs'])

    # Extract the first row (initial state) with positions and velocities
    initial_state = (df['q1'][0], df['v1'][0], df['q2'][0], df['v2'][0])
    initial_states.append(initial_state)  # Store the initial state

    errors = df['Terminal_Cost_Error']
    error.append(errors)

# Create a figure with two subplots in a single row
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns of plots

# Plot Terminal Costs in the first subplot
for i, tc in enumerate(terminal_costs):
    label = f'q0={initial_states[i][0]:.4f}, v0={initial_states[i][1]:.4f}'
    axes[0].plot(tc, label=label)  # Plot each test's data

axes[0].set_title("Learned Terminal Cost")
axes[0].set_xlabel('Simulation steps')       # Set x-axis label
axes[0].set_ylabel(r'$\hat{V}_1(x_1)$', fontsize = 12) # Set y-axis label
axes[0].legend()                     # Show legend
axes[0].grid(True)

# Plot True Terminal Costs in the second subplot
for i, ttc in enumerate(true_terminal_costs):
    label = f'q0={initial_states[i][0]:.4f}, v0={initial_states[i][1]:.4f}'
    axes[1].plot(ttc, label=label)  # Plot each test's data
axes[1].set_xlabel('Simulation steps')            # Set x-axis label
axes[1].set_ylabel(r'$V_1(x_1)$', fontsize = 12) # Set y-axis label
axes[1].legend()                          # Show legend
axes[1].grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
for i, err in enumerate(error):
    label = f'q0={initial_states[i][0]:.4f}, v0 = {initial_states[i][1]:.4f}'
    plt.plot(err, label = label)
plt.xlabel('Simulation steps')
plt.ylabel(r'$e = \hat{V}_1(x_1) - V_1(x_1)$')
plt.grid(True)
plt.legend()
plt.show()