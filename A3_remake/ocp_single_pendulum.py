import casadi as ca
import numpy as np
import pandas as pd

# Define the OCP variables
T = 5.0  # Time horizon
dt = 0.01  # Time step
N = T/dt  # Number of control intervals

# Define the system dynamics
def dynamics(x, u):
    l = 1               # lenght of link
    m = 1               # mass of link
    g = 9.81            # gravity

    # State extraction from parameters
    q, dq = x[0], x[1]

    ddq = -g/l * ca.sin(q) + 1/(m * l**2) * u

    x_next = ca.vertcat(q + dt * dq, dq + dt * ddq)

    return  x_next

# Create an optimization problem
ocp = ca.Opti()

# Define the decision variables
q = ocp.variable(N+1)  # State trajectory [theta, theta_dot]
v = ocp.variable(N+1)
u = ocp.variable(N)      # Control trajectory

# Define the initial and final states
# Set the initial state constraints
x_init = ocp.parameter(2)  # Initial state [theta_init, theta_dot_init]
# Initial state constraint
ocp.subject_to(q[0] == x_init[0])
ocp.subject_to(v[0] == x_init[1])

# Set the dynamics constraints
for k in range(N):
  x_next = dynamics(np.array(q[k], v[k]), u[k])
  ocp.subject_to(q[k+1] == x_next[0])
  ocp.subject_to(v[k+1] == x_next[1])

# Define the cost function
w_q = 1e2
w_v = 1e-1
w_u = 1e-4

cost = 0.0
for k in range(N):
  cost += w_v * v[k]**2 + w_u * u[k]**2
ocp.minimize(cost)

# Set the bounds on the control variable
ocp.subject_to(-9.81 <= u <= 9.81)
# Set the state constraints
theta_min = 3/4*ca.pi  # Minimum pendulum angle
theta_max = 5/4*ca.pi   # Maximum pendulum angle
theta_dot_min = -10  # Minimum pendulum angular velocity
theta_dot_max = 10   # Maximum pendulum angular velocity

for k in range(N+1):
  ocp.subject_to(theta_min <= q[k] <= theta_max)  # Pendulum angle constraint
  ocp.subject_to(theta_dot_min <= v[k] <= theta_dot_max)  # Pendulum angular velocity constraint

# Set the solver options
opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-3}
ocp.solver("ipopt", opts)

# Define the grid for initial state sampling
theta_init_grid = np.linspace(theta_min, theta_max, num=121)
theta_dot_init_grid = np.linspace(theta_dot_min, theta_dot_max, num=121)

# Create buffers to store initial states and terminal costs
initial_states = []
terminal_costs = []

# Solve the OCP for each initial state in the grid
for theta_init in theta_init_grid:
  for theta_dot_init in theta_dot_init_grid:
    x_init_value = np.array([theta_init, theta_dot_init])
    ocp.set_value(x_init, x_init_value)
    sol = ocp.solve()
    x_opt = sol.value(np.array(q,v))
    u_opt = sol.value(u)
    # Process the optimal solution for each initial state
    
    # Save the initial state and terminal cost in the buffer
    initial_states.append(x_init_value)
    terminal_costs.append(sol.value(cost))

df = pd.DataFrame({'Initial State': initial_states, 'Terminal Cost': terminal_costs})

# Save the dataframe to a CSV file
df.to_csv('results.csv', index=False)
