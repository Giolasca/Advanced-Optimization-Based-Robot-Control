import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
data = pd.read_csv('ocp_data_SP.csv')

# Extract the necessary columns
q1 = data['q1']
v1 = data['v1']
costo = data['Costs']


# 2D scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
scatter = ax.scatter(q1, v1, c=costo, cmap='viridis', alpha=0.6)

# Color bar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, aspect=5)
cbar.set_label('Cost')

# Set axis labels
ax.set_xlabel('q1')
ax.set_ylabel('v1')
plt.title('2D Plot of q1 and v1 with Cost Color Bar')

# Show the plot
plt.show()


# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
p = ax.scatter(q1, v1, costo, c=costo, cmap='viridis', alpha=0.6)

# Set axis labels
ax.set_xlabel('q1')
ax.set_ylabel('v1')
ax.set_zlabel('Cost')
plt.title('3D Plot of q1, v1, and Cost')

# Show the plot
plt.show()