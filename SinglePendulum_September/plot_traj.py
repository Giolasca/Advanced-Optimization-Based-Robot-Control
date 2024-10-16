import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_trajectory_from_csv(filename):
    """
    Load trajectory (position, velocity, and input) from a CSV file.
    """
    return pd.read_csv(filename)

def plot_trajectories_with_reference(file1, file2, file3, reference_position, reference_velocity, reference_input):
    """
    Load trajectories from two CSV files and plot them on a single graph along with reference values.
    """
    # Load data from CSV
    traj1 = load_trajectory_from_csv(file1)
    traj2 = load_trajectory_from_csv(file2)
    traj3 = load_trajectory_from_csv(file3)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Position plot
    axs[0].plot(traj1['Positions'], label="Trajectory with terminal cost (N=5) and T = 0.01", color='b', linestyle='-')  # Solid line
    axs[0].plot(traj2['Positions'], label="Trajectory without terminal cost (N=50) and T = 1", color='r', linestyle=':')  # Dotted line
    axs[0].plot(traj3['Positions'], label="Trajectory without terminal cost (N=50) and T = 0.01", color='m', linestyle='-.')  # Dash-dot line
    axs[0].plot(reference_position, label="Reference", color='g', linestyle='--')  # Dashed line for the reference
    axs[0].set_title('Position Comparison')
    axs[0].set_ylabel('Position')
    axs[0].legend(loc='best')
    axs[0].grid(True)
    
    # Velocity plot
    axs[1].plot(traj1['Velocities'], label="Trajectory with terminal cost (N=5) and T = 0.01", color='b', linestyle='-')  # Solid line
    axs[1].plot(traj2['Velocities'], label="Trajectory without terminal cost (N=50) and T = 1", color='r', linestyle=':')  # Dotted line
    axs[1].plot(traj3['Velocities'], label="Trajectory without terminal cost (N=50) and T = 0.01", color='m', linestyle='-.')  # Dash-dot line
    axs[1].plot(reference_velocity, label="Reference", color='g', linestyle='--')  # Dashed line for the reference
    axs[1].set_title('Velocity Comparison')
    axs[1].set_ylabel('Velocity')
    axs[1].legend(loc='best')
    axs[1].grid(True)
    
    # Input plot
    axs[2].plot(traj1['Inputs'], label="Trajectory with terminal cost (N=5) and T = 0.01", color='b', linestyle='-')  # Solid line
    axs[2].plot(traj2['Inputs'], label="Trajectory without terminal cost (N=50) and T = 1", color='r', linestyle=':')  # Dotted line
    axs[2].plot(traj3['Inputs'], label="Trajectory without terminal cost (N=50) and T = 0.01", color='m', linestyle='-.')  # Dash-dot line
    axs[2].set_title('Input Comparison')
    axs[2].set_ylabel('Input')
    axs[2].set_xlabel('Time Steps')
    axs[2].legend(loc='best')
    axs[2].grid(True)

    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    # Read files .csv
    csv_file1 = 'Plots_&_Animations/mpc_SP_NTC_225_unconstr.csv'
    csv_file2 = 'Plots_&_Animations/mpc_SP_TC_225_unconstr.csv'
    csv_file3 = 'Plots_&_Animations/mpc_SP_NTC_T_0.01_225_unconstr.csv'
    
    # Reference definitions
    reference_value = (5/4) * np.pi  # Reference constant for position
    time_steps = 50  # Number of time steps
    
    reference_position = [reference_value] * time_steps  # Constant reference position
    reference_velocity = [0] * time_steps  # Constant reference velocity
    reference_input = [0] * time_steps     # Constant control input
    
    # Execute the plot with references
    plot_trajectories_with_reference(csv_file1, csv_file2, csv_file3, reference_position, reference_velocity, reference_input)

