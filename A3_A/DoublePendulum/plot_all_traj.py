import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_trajectories(output_dir, num_tests):
    """
    Reads trajectories from each test and plots them in separate graphs.
    
    :param output_dir: Directory where the results of each test are saved.
    :param num_tests: Total number of tests executed.
    """
    
    # Initialize lists to collect the trajectories
    all_q1 = []
    all_q2 = []
    all_v1 = []
    all_v2 = []
    all_u1 = []
    all_u2 = []
    
    # Loop through the generated files for each test
    for test_idx in range(1, num_tests + 1):
        test_dir = os.path.join(output_dir, f'Test_180_180_unconstr_test_{test_idx}')
        
        # Search for the CSV file for the current test
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
        if len(test_files) == 0:
            print(f"No CSV file found for test {test_idx}")
            continue  # Skip to the next test if no CSV file is found

        # Assume there is only one CSV file per test
        csv_file = os.path.join(test_dir, test_files[0])
        
        # Read the CSV file as a dataframe
        df = pd.read_csv(csv_file)

        # Assume the columns of the CSV are 'q1', 'q2', 'v1', 'v2', 'u1', 'u2'
        q1 = df['q1'].values
        q2 = df['q2'].values
        v1 = df['v1'].values
        v2 = df['v2'].values
        u1 = df['u1'].values
        u2 = df['u2'].values

        # Append the trajectory to the lists
        all_q1.append(q1)
        all_q2.append(q2)
        all_v1.append(v1)
        all_v2.append(v2)
        all_u1.append(u1)
        all_u2.append(u2)

    # Create the first plot for q1 and q2
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    for q in all_q1:
        plt.plot(q)
    plt.xlabel('MPC Step')  
    plt.ylabel('q1 [rad]')  
    plt.title('q1 Over All Tests')  
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for q in all_q2:
        plt.plot(q)
    plt.xlabel('MPC Step')  
    plt.ylabel('q2 [rad]')  
    plt.title('q2 Over All Tests')  
    plt.grid(True)

    # Save and show the first plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q1_q2_plot.png'))
    plt.show()

    # Create the second plot for v1 and v2
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    for v in all_v1:
        plt.plot(v)
    plt.xlabel('MPC Step')  
    plt.ylabel('v1 [rad/s]')  
    plt.title('v1 Over All Tests')  
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for v in all_v2:
        plt.plot(v)
    plt.xlabel('MPC Step')  
    plt.ylabel('v2 [rad/s]')  
    plt.title('v2 Over All Tests')  
    plt.grid(True)

    # Save and show the second plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v1_v2_plot.png'))
    plt.show()

    # Create the third plot for u1 and u2
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    for u in all_u1:
        plt.plot(u)
    plt.xlabel('MPC Step')  
    plt.ylabel('u1 [rad/s^2]')  
    plt.title('u1 Over All Tests')  
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for u in all_u2:
        plt.plot(u)
    plt.xlabel('MPC Step')  
    plt.ylabel('u2 [rad/s^2]')  
    plt.title('u2 Over All Tests')  
    plt.grid(True)

    # Save and show the third plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'u1_u2_plot.png'))
    plt.show()

# Example usage:
output_dir = 'Plots_&_Animations'  
num_tests = 15  # Total number of tests executed
plot_all_trajectories(output_dir, num_tests)  