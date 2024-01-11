import numpy as np
import ocp_double_pendulum_conf as conf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Load data from CSV
data = pd.read_csv("double_data.csv")

# Filter data based on the 'viable' column
viable_states = data[data['viable_states'] == 1][['q1', 'v1', 'q2', 'v2']].values
non_viable_states = data[data['viable_states'] == 0][['q1', 'v1', 'q2', 'v2']].values

# Check if viable_states is not empty before concatenating
if viable_states.size == 0:
    all_states = non_viable_states if non_viable_states.size != 0 else np.array([])  
else:
    # Check if non_viable_states is not empty before concatenating
    if non_viable_states.size == 0:
        all_states = viable_states
    else:
        # Merge data from viable_states and non_viable_states
        all_states = np.vstack((viable_states, non_viable_states))

# Create a color array to distinguish between viable and no_viable states
colors = ['red'] * len(viable_states) + ['blue'] * len(non_viable_states)


# Function to handle click on the first plot
def on_first_plot_click(event):
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        x_apprx = round(x, 3)
        y_apprx = round(y, 3)
        second_plot(all_states, x_apprx, y_apprx, colors)


# Function for the second plot
def second_plot(all_states, x, y, colors, tolerance=0.3):
    fig, ax = plt.subplots()
    ax.set_title(f'Second Plot - Selected Point: ({x}, {y})')
    ax.set_xlabel('q2 [rad]')
    ax.set_ylabel('dq2 [rad/s]')

    # Calculate Euclidean distance between the clicked point and all points in q1_v1
    distances = np.linalg.norm(all_states[:, :2] - np.array([x, y]), axis=1)

    # Filter points based on tolerance
    selected_point_indices = np.where(distances < tolerance)[0]
    q2_v2 = all_states[selected_point_indices, 2:]

    # Generate all possible combinations of q2 and v2
    q2_values, v2_values = np.unique(q2_v2[:, 0]), np.unique(q2_v2[:, 1])

    # Create all possible combinations of q2 and v2
    combinations = np.array(np.meshgrid(q2_values, v2_values)).T.reshape(-1, 2)

    # Plot points in the second plot with colors based on viability
    viable_mask = selected_point_indices < len(viable_states)
    ax.scatter(combinations[viable_mask, 0], combinations[viable_mask, 1], c='red', label='viable')
    ax.scatter(combinations[~viable_mask, 0], combinations[~viable_mask, 1], c='blue', label='no_viable')

    ax.legend()
    plt.show()


# First plot q1-v1 pairs
fig, ax = plt.subplots()
ax.set_title('First Plot')
ax.set_xlabel('q1 [rad]')
ax.set_ylabel('dq1 [rad/s]')

# Plot points for q1 and v1 with colors based on viability
if len(viable_states) != 0:
    ax.scatter(viable_states[:,0], viable_states[:,1], c='r', label='viable')
    ax.legend()
if len(non_viable_states) != 0:
    ax.scatter(non_viable_states[:,0], non_viable_states[:,1], c='b', label='non-viable')
    ax.legend()

ax.legend()
fig.canvas.mpl_connect('button_press_event', on_first_plot_click)
plt.show()