import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

# Load data from .mat file
mat_data = scipy.io.loadmat('viable.mat')

# Assuming position is in the first column (column 0) and velocity in the second column (column 1)
positions = mat_data['viable_states'][:, 0:2]  # Replace 'your_data' with the actual variable name

# Plot the data
plt.scatter(positions[:, 0], positions[:, 1], label='Data Points')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Pendulum State Space Representation')
plt.grid(True)
plt.legend()

# Fit an ellipse to the data
def ellipse_func(x, x0, y0, a, b, theta):
    """
    Ellipse function in the form: (x-x0)**2/a**2 + (y-y0)**2/b**2 = 1
    """
    return (
        (np.cos(theta) * (x - x0) - np.sin(theta) * (y0))**2 / a**2 +
        (np.sin(theta) * (x - x0) + np.cos(theta) * (y0))**2 / b**2 - 1
    )

# Initial guess for the ellipse parameters
initial_guess = [np.pi, 0, np.pi, 2, -np.pi]
eps = 1e-6
# Fixing the center of the ellipse
fixed_center = [np.pi, 0]  # Replace with the desired center coordinates
bounds = ([fixed_center[0]-eps, fixed_center[1]-eps, 0, 0, -np.pi], [fixed_center[0]+eps, fixed_center[1]+eps, np.pi, 5, np.pi])

# Fit the ellipse using curve_fit with fixed center
fit_params, _ = curve_fit(ellipse_func, positions[:, 0], positions[:, 1], p0 = initial_guess, bounds = bounds, method = 'trf')

# Plot the fitted ellipse
fitted_ellipse = Ellipse((fit_params[0], fit_params[1]), fit_params[2] * 2, fit_params[3] * 2, angle=np.radians(fit_params[4]), edgecolor='r', fc='None', lw=2, label='Fitted Ellipse')
plt.gca().add_patch(fitted_ellipse)

plt.legend()
plt.show()

# Display the ellipse parameters
print("Fitted Ellipse Parameters:")
print("Center (x0, y0):", fit_params[0], fit_params[1])
print("Semi-Major Axis (a):", fit_params[3])
print("Semi-Minor Axis (b):", fit_params[2])
print("Rotation Angle (theta):", np.degrees(fit_params[4]))
