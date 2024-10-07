import numpy as np
import random

### Execution methods
multiproc = 1
num_processes = 4  # Number of processor
grid = 1

### Horizon parameters
T = 1.0          # OCP horizon size
dt = 0.01        # OCP time step
N = int(T/dt);   # Number of horizon step
max_iter = 100   # Maximum iteration per point


### Constaints for the first link###
q1_min = 3/4*np.pi
q1_max = 5/4*np.pi
v1_min = -10
v1_max = 10

'''
u1_min = -9.81
u1_max = 9.81
'''

### WWeights for the first link ###
w_q1 = 1e2
w_v1 = 1e-1
w_u1 = 1e-4


### Constaints for the second link ###
q2_min = 3/4*np.pi
q2_max = 5/4*np.pi
v2_min = -10
v2_max = 10

'''
u2_min = -9.81
u2_max = 9.81
'''

### Weights for the second link###
w_q2 = 1e2
w_v2 = 1e-1
w_u2 = 1e-4

### To spit the state_array
start_index = 21*21*21*4   
end_index = 21*21*21*7    
tot_points = 21*21*21*21

q1_target = np.pi
q2_target = np.pi

# Function to create states array in a grid
def grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2):
    n_ics = n_pos_q1 * n_pos_q2 * n_vel_v1 * n_vel_v2
    possible_q1 = np.linspace(q1_min, q1_max, num=n_pos_q1)
    possible_v1 = np.linspace(v1_min, v1_max, num=n_vel_v1)
    possible_q2 = np.linspace(q2_min, q2_max, num=n_pos_q2)
    possible_v2 = np.linspace(v2_min, v2_max, num=n_vel_v2)
    state_array = np.zeros((n_ics, 4))

    i = 0
    for q1 in possible_q1:
        for v1 in possible_v1:
            for q2 in possible_q2:
                for v2 in possible_v2:
                    state_array[i, :] = np.array([q1, v1, q2, v2])
                    i += 1

    state_array = state_array[start_index:end_index]
    return state_array


# Function to create states array taken from a uniform distribution
def random_states(n_q1v1):
    state_array = np.zeros((n_q1v1 * 441, 4))

    for i in range(n_q1v1):
        # Generate a random pair of q1 and v1
        q1 = (q1_min - q1_max) * np.random.random_sample() + q1_min
        v1 = (v1_min - v1_max) * np.random.random_sample() + v1_min

        for j in range(441):
            # Generate random pairs of q2 and v2 for the random pair of q1 and v1
            q2 = (q2_min - q2_max) * np.random.random_sample() + q2_min
            v2 = (v2_min - v2_max) * np.random.random_sample() + v2_min
            state_array[i * 441 + j, :] = np.array([q1, v1, q2, v2])

    return state_array