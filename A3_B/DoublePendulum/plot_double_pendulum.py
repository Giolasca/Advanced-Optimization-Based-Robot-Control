import numpy as np
import mpc_double_pendulum_conf as conf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

positions = pd.read_csv("../DoublePendulum/Plots_&_Animations/Position_double.csv")
positions_q1 = np.array(positions['Positions_q1'])
positions_q2 = np.array(positions['Positions_q2'])

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.grid()

line_q1, = ax.plot([], [], 'o-', lw=4)
line_q2, = ax.plot([], [], 'o-', lw=4)
trace_q1, = ax.plot([], [], '.-', lw=0, ms=2, color='red')
trace_q2, = ax.plot([], [], '.-', lw=0, ms=2, color='blue')
time_template = 'time = %.2f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# axis limits
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-0.1, 2.1])
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)


# bounds for link1 
x1_line1 = np.linspace(0.7, 0, 50)
y1_line1 = np.tan(np.radians(45)) * x1_line1
ax.plot(x1_line1, y1_line1, color='blue', linestyle='--', lw=2)

x1_line2 = np.linspace(-0.7, 0, 50)
y1_line2 = np.tan(np.radians(135)) * x1_line2
ax.plot(x1_line2, y1_line2, color='blue', linestyle='--', lw=2)


def animate(i):
    thisx_q1 = [0, np.sin(positions_q1[i])]
    thisy_q1 = [0, -np.cos(positions_q1[i])]

    thisx_q2 = [np.sin(positions_q1[i]), np.sin(positions_q1[i]) + np.sin(positions_q2[i])]
    thisy_q2 = [-np.cos(positions_q1[i]), -np.cos(positions_q1[i]) - np.cos(positions_q2[i])]

    history_x_q1 = np.sin(positions_q1[:i])
    history_y_q1 = -np.cos(positions_q1[:i])

    history_x_q2 = np.sin(positions_q1[:i]) + np.sin(positions_q2[:i])
    history_y_q2 = -np.cos(positions_q1[:i]) - np.cos(positions_q2[:i])

    line_q1.set_data(thisx_q1, thisy_q1)
    line_q2.set_data(thisx_q2, thisy_q2)
    trace_q1.set_data(history_x_q1, history_y_q1)
    trace_q2.set_data(history_x_q2, history_y_q2)
    time_text.set_text(time_template % (i * conf.dt))

    return line_q1, line_q2, trace_q1, trace_q2, time_text

ani = animation.FuncAnimation(fig, animate, len(positions_q1), interval=50, blit=True)

# Save the animation as a GIF file using Pillow
animation_file_path = "../DoublePendulum/Plots_&_Animations/Double_Pendulum.gif"
ani.save(animation_file_path, writer='pillow', fps=20)

# Show the animation
plt.show()

