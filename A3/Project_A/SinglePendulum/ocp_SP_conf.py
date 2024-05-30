import numpy as np

multiproc = 1
num_processes = 4
grid = 1

T = 1.0          # OCP horizon size
dt = 0.01        # OCP time step
N = int(T/dt);   # Number of horizon step

### Constaints for the pendulum ###
q_min = 3/4*np.pi
q_max = 5/4*np.pi
v_min = -10
v_max = 10
u_min = -9.81
u_max = 9.81

### Weights for the pendulum ###
w_q = 1e2
w_v = 1e-1
w_u = 1e-4


# Function to create states array in a grid
def grid_states(n_pos, n_vel):
    n_ics = n_pos * n_vel
    possible_q = np.linspace(q_min, q_max, num=n_pos)
    possible_v = np.linspace(v_min, v_max, num=n_vel)
    state_array = np.zeros((n_ics, 2))

    i = 0
    for q in possible_q:
        for v in possible_v:
            state_array[i, :] = np.array([q, v])
            i += 1

    return state_array


# Function to create states array taken from a uniform distribution
def random_states(n_states):
    state_array = np.zeros((n_states, 2))

    for i in range(n_states):
        possible_q = (q_max - q_min) * np.random.random_sample() + q_min
        possible_v = (v_max - v_min) * np.random.random_sample() + v_min
        state_array[i,:] = np.array([possible_q, possible_v])
    
    return n_states, state_array