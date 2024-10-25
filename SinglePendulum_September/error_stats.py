import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of CSV files for the 15 tests
csv_files = [
    'Plots_&_Animations/Test_225_unconstr_tanh_test_1/mpc_SP_TC_225_unconstr_tanh_test_1.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_2/mpc_SP_TC_225_unconstr_tanh_test_2.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_3/mpc_SP_TC_225_unconstr_tanh_test_3.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_4/mpc_SP_TC_225_unconstr_tanh_test_4.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_5/mpc_SP_TC_225_unconstr_tanh_test_5.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_6/mpc_SP_TC_225_unconstr_tanh_test_6.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_7/mpc_SP_TC_225_unconstr_tanh_test_7.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_8/mpc_SP_TC_225_unconstr_tanh_test_8.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_9/mpc_SP_TC_225_unconstr_tanh_test_9.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_11/mpc_SP_TC_225_unconstr_tanh_test_11.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_12/mpc_SP_TC_225_unconstr_tanh_test_12.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_13/mpc_SP_TC_225_unconstr_tanh_test_13.csv',
    'Plots_&_Animations/Test_225_unconstr_tanh_test_15/mpc_SP_TC_225_unconstr_tanh_test_15.csv'
]

# Initialize lists to store statistics
means = []
medians = []
std_devs = []
error_data = []

# Plot configuration for histograms
plt.figure(figsize=(15, 10))
num_tests = len(csv_files)
num_rows = (num_tests + 2) // 3  # Determine number of subplot rows
num_cols = min(num_tests, 3)     # Max 3 columns

# Iterate over each CSV file and process the error data
for i, file in enumerate(csv_files):
    if not os.path.exists(file):
        print(f"File {file} not found.")
        continue

    # Load the data
    data = pd.read_csv(file)
    
    # Assuming the error column is named 'Terminal_Cost_Error'
    error = data['Terminal_Cost_Error']

    # Remove NaN values from the error data before calculating statistics
    error = error.dropna()

    # Calculate statistics
    mean_error = np.mean(error)
    median_error = np.median(error)
    std_dev_error = np.std(error)

    # Append statistics to lists
    means.append(mean_error)
    medians.append(median_error)
    std_devs.append(std_dev_error)
    error_data.append(error)  # Store the error data for further plots

    # Plot histogram for the current test
    plt.subplot(num_rows, num_cols, i + 1)
    plt.hist(error, bins=20, alpha=0.75, color='b', edgecolor='black')
    plt.title(f"Test {i+1}: Error Distribution")
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)

# Show all histograms
plt.tight_layout()
plt.show()

# Summary of statistics
print("Summary of Statistics (mean, median, std deviation) for each test:")
for i in range(num_tests):
    print(f"Test {i+1}: Mean = {means[i]:.4f}, Median = {medians[i]:.4f}, Std Dev = {std_devs[i]:.4f}")

# --- Additional Visualizations ---

# Plot 1: Box Plot of Errors Across All Tests
plt.figure(figsize=(10, 6))
plt.boxplot(error_data, patch_artist=True)
plt.title('Error Distribution Across Tests (Box Plot)')
plt.xlabel('Test Number')
plt.ylabel('Error')
plt.grid(True)
plt.show()

# Plot 2: Bar Plot of Summary Statistics (Mean and Std Dev)
tests = np.arange(1, len(csv_files) + 1)

plt.figure(figsize=(12, 6))
plt.bar(tests, means, color='blue', alpha=0.6, label='Mean Error')
plt.errorbar(tests, means, yerr=std_devs, fmt='o', color='black', label='Standard Deviation', capsize=5)
plt.axhline(y=np.median(means), color='red', linestyle='--', label='Median of Means')

plt.title('Error Statistics Across Tests (Mean and Std Dev)')
plt.xlabel('Test Number')
plt.ylabel('Error Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Heatmap of Errors Across Tests
error_matrix = np.array([pd.read_csv(file)['Terminal_Cost_Error'].dropna().values for file in csv_files])

plt.figure(figsize=(10, 6))
sns.heatmap(error_matrix, cmap="coolwarm", annot=False)
plt.title('Heatmap of Error Distribution Across Tests')
plt.xlabel('Timestep')
plt.ylabel('Test Number')
plt.show()

# Plot 4: Scatter Plot of Mean Error vs Standard Deviation
plt.figure(figsize=(8, 6))
plt.scatter(means, std_devs, color='green', s=100, alpha=0.75)
plt.title('Mean Error vs Standard Deviation Across Tests')
plt.xlabel('Mean Error')
plt.ylabel('Standard Deviation of Error')
plt.grid(True)
plt.show()

# Plot 5: Cumulative Error Plot
plt.figure(figsize=(10, 6))

for i, file in enumerate(csv_files):
    error = pd.read_csv(file)['Terminal_Cost_Error'].dropna()
    sorted_error = np.sort(error)
    cdf = np.arange(len(sorted_error)) / len(sorted_error)
    plt.plot(sorted_error, cdf, label=f'Test {i+1}')

plt.title('Cumulative Distribution of Errors Across Tests')
plt.xlabel('Error')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()