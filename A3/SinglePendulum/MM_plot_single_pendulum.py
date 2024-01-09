import numpy as np
import MM_mpc_single_pendulum_conf as conf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

positions = np.array(pd.read_csv("../SinglePendulum/Plots_&_Animations/P1_Position.csv").values.tolist()).flatten()

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=4)
trace, = ax.plot([], [], '.-', lw=0, ms=2, color='red')
time_template = 'time = %.2f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# axis limits
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)

# bounds
x_line1 = np.linspace(0.7, 0, 50)
y_line1 = np.tan(np.radians(45)) * x_line1
ax.plot(x_line1, y_line1, color='blue', linestyle='--', lw=2)

x_line2 = np.linspace(-0.7, 0, 50)
y_line2 = np.tan(np.radians(135)) * x_line2
ax.plot(x_line2, y_line2, color='blue', linestyle='--', lw=2)

# Set grid with equidistant spacing
ax.set_xticks(np.arange(-1.1, 1.1, 0.2))
ax.set_yticks(np.arange(-0.1, 1.2, 0.2))

def animate(i):
    thisx = [0, np.sin(positions[i])]
    thisy = [0, -np.cos(positions[i])]

    history_x = np.sin(positions[:i])
    history_y = -np.cos(positions[:i])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*conf.dt))
    return line, trace, time_text

ani = animation.FuncAnimation(fig, animate, len(positions), interval=50, blit=True)

# Save the animation as a GIF file using Pillow
animation_file_path = "../SinglePendulum/Plots_&_Animations/P1_Pendulum.gif"
ani.save(animation_file_path, writer='pillow', fps=20)

# Show the animation
plt.show()
