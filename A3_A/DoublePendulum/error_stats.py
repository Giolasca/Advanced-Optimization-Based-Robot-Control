import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Parameters
num_tests = 15
noise = 1

if noise:
    base_path = 'Plots_&_Animations/Test_180_180_unconstr_test_{}/mpc_DP_TC_noise_180_180_unconstr_test_{}.csv'
else:
    base_path = 'Plots_&_Animations/Test_180_180_unconstr_test_{}/mpc_DP_TC_180_180_unconstr_test_{}.csv'


# Initialize lists for statistics and data
means = []
medians = []
std_devs = []
error_data = []

# Set up layout for histograms
plt.figure(figsize=(15, 10))
num_rows = (num_tests + 2) // 3  # Calculate the number of rows for subplots
num_cols = min(num_tests, 3)     # Max of 3 columns

# Loop through each test, calculating statistics
for test_number in range(1, num_tests + 1):
    file_path = base_path.format(test_number, test_number)

    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        continue

    # Load data
    data = pd.read_csv(file_path)
    error = data['Terminal_Cost_Error'].dropna()

    # Calculate statistics
    mean_error = np.mean(error)
    median_error = np.median(error)
    std_dev_error = np.std(error)

    # Append statistics to lists
    means.append(mean_error)
    medians.append(median_error)
    std_devs.append(std_dev_error)
    error_data.append(error)

    # Histogram for the current test
    plt.subplot(num_rows, num_cols, test_number)
    plt.hist(error, bins=20, alpha=0.75, color='b', edgecolor='black')
    plt.title(f"Test {test_number}: Error Distribution")
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Summary of statistics
print("Summary of Statistics (mean, median, std deviation) for each test:")
for i, (mean, median, std_dev) in enumerate(zip(means, medians, std_devs), 1):
    print(f"Test {i}: Mean = {mean:.4f}, Median = {median:.4f}, Std Dev = {std_dev:.4f}")

# --- Additional Visualizations ---

# 1. Box Plot of errors across all tests
plt.figure(figsize=(10, 6))
plt.boxplot(error_data, patch_artist=True)
plt.title('Error Distribution Across Tests (Box Plot)')
plt.xlabel('Test Number')
plt.ylabel('Error')
plt.grid(True)
plt.show()

# 2. Bar Plot of summary statistics
tests = np.arange(1, len(means) + 1)
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

# 3. Heatmap of errors across tests per timestep
max_length = max(map(len, error_data))
error_matrix = np.array([np.pad(e, (0, max_length - len(e)), constant_values=np.nan) for e in error_data])
plt.figure(figsize=(10, 6))
sns.heatmap(error_matrix, cmap="coolwarm", annot=False, mask=np.isnan(error_matrix))
plt.title('Heatmap of Error Distribution Across Tests')
plt.xlabel('Timestep')
plt.ylabel('Test Number')
plt.show()

# 4. Scatter Plot of Mean Error vs Standard Deviation
plt.figure(figsize=(8, 6))
plt.scatter(means, std_devs, color='green', s=100, alpha=0.75)
plt.title('Mean Error vs Standard Deviation Across Tests')
plt.xlabel('Mean Error')
plt.ylabel('Standard Deviation of Error')
plt.grid(True)
plt.show()

# 5. Cumulative Error Plot
plt.figure(figsize=(10, 6))
for i, error in enumerate(error_data, 1):
    sorted_error = np.sort(error)
    cdf = np.arange(len(sorted_error)) / len(sorted_error)
    plt.plot(sorted_error, cdf, label=f'Test {i}')

plt.title('Cumulative Distribution of Errors Across Tests')
plt.xlabel('Error')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()
