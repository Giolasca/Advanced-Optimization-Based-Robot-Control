import numpy as np

multiproc = 1
num_processes = 4
grid = 1

T = 0.5                   # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

### Constaints for the pendulum ###
lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
lowerVelocityLimit = -10
upperVelocityLimit = 10
lowerControlBound = -9.81
upperControlBound = 9.81

### Weights for the pendulum ###
w_q = 1e2
w_v = 1e-1
w_u = 1e-4


# Function to create states array in a grid
def grid_states(n_pos, n_vel):
    n_ics = n_pos * n_vel
    possible_q = np.linspace(lowerPositionLimit, upperPositionLimit, num=n_pos)
    possible_v = np.linspace(lowerVelocityLimit, upperVelocityLimit, num=n_vel)
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
        possible_q = (upperPositionLimit - lowerPositionLimit) * np.random.random_sample() + lowerPositionLimit
        possible_v = (upperVelocityLimit - lowerVelocityLimit) * np.random.random_sample() + lowerVelocityLimit
        state_array[i,:] = np.array([possible_q, possible_v])
    
    return n_states, state_array