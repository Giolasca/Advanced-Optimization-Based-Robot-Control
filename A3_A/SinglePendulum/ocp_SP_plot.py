import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
data = pd.read_csv('ocp_data_SP_target_225_unconstr.csv')

# Extract the necessary columns
q1 = data['position']
v1 = data['velocity']
costo = data['cost']


# 2D scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
scatter = ax.scatter(q1, v1, c=costo, cmap='viridis', alpha=0.6)

# Color bar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, aspect=5)
cbar.set_label('cost')

# Set axis labels
ax.set_xlabel('position')
ax.set_ylabel('velocity')
plt.title('2D Plot of q1 and v1 with Cost Color Bar')

# Show the plot
plt.show()


# 3D plot
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
p = ax.scatter(q1, v1, costo, c=costo, cmap='viridis', alpha=0.6)

# Set axis labels
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('cost')
plt.title('3D Plot of q1, v1, and Cost')

# Show the plot
plt.show()