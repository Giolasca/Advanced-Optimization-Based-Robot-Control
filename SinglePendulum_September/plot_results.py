import pandas as pd
import matplotlib.pyplot as plt

# Function to load data from the CSV files
def load_data(file1, file2):
    # Load the first CSV containing True_Terminal_Costs and V_hat
    data1 = pd.read_csv(file1)
    true_terminal_costs = data1['True_Terminal_Costs']
    err_ext = data1['Terminal_Cost_Error']  

    # Load the second CSV containing Total_Costs
    data2 = pd.read_csv(file2)
    total_costs = data2['Total_Costs']

    return true_terminal_costs, err_ext, total_costs

def plot_costs_and_error(true_terminal_costs, total_costs, err_ext):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot: True_Terminal_Costs and Total_Costs
    axs[0].plot(true_terminal_costs, label='LTC-MPC with N = 1', color='blue')
    axs[0].plot(total_costs, label='Full MPC with N = 100', color='red')
    axs[0].set_ylabel('Cost-to-go')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    # Second subplot: External Error (err_ext)
    axs[1].plot(err_ext, label='Error (err_ext)', color='blue', marker = 'o')
    axs[1].set_xlabel('Simulation steps')
    axs[1].set_ylabel(r'$e = \hat{V}_1 - V_1$', fontsize=12)  
    axs[1].legend(loc='best')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# CSV files
file2 = 'Plots_&_Animations/mpc_SP_NTC_T_1_225_unconstr_tanh.csv'  
file1 = 'Plots_&_Animations/mpc_SP_TC_225_unconstr_tanh.csv'  

# Load the data
true_terminal_costs, err_ext, total_costs = load_data(file1, file2)

# Create the plot
plot_costs_and_error(true_terminal_costs, total_costs, err_ext)