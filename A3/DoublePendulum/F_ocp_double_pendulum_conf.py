import numpy as np
import random

multiproc = 1
num_processes = 4

grid = 1           # Select grid or random state_array
old_data = 1       
save_data = 1      

T = 1          # OCP horizion
dt = 0.01        # OCP time step
max_iter = 100     # Maximum iteration per point

### Constaints for the first link ###
lowerPositionLimit_q1 = 3/4*np.pi
upperPositionLimit_q1 = 5/4*np.pi
lowerVelocityLimit_v1 = -10
upperVelocityLimit_v1 = 10
lowerControlBound_u1 = -9.81
upperControlBound_u1 = 9.81

### Weights for the first link ###
w_q1 = 1e2
w_v1 = 1e-1
w_u1 = 1e-4


### Constaints for the second link ###
lowerPositionLimit_q2 = 3/4*np.pi
upperPositionLimit_q2 = 5/4*np.pi
lowerVelocityLimit_v2 = -10
upperVelocityLimit_v2 = 10
lowerControlBound_u2 = -9.81
upperControlBound_u2 = 9.81

### Weights for the second link ###
w_q2 = 1e2
w_v2 = 1e-1
w_u2 = 1e-4


# Function to create states array in a grid
def grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2):
    n_ics = n_pos_q1 * n_pos_q2 * n_vel_v1 * n_vel_v2
    possible_q1 = np.linspace(lowerPositionLimit_q1, upperPositionLimit_q1, num=n_pos_q1)
    possible_v1 = np.linspace(lowerVelocityLimit_v1, upperVelocityLimit_v1, num=n_vel_v1)
    possible_q2 = np.linspace(lowerPositionLimit_q2, upperPositionLimit_q2, num=n_pos_q2)
    possible_v2 = np.linspace(lowerVelocityLimit_v2, upperVelocityLimit_v2, num=n_vel_v2)
    state_array = np.zeros((n_ics, 4))
    
    i = 0
    for q1 in possible_q1:
        for v1 in possible_v1:
            for q2 in possible_q2:
                for v2 in possible_v2:
                    state_array[i, :] = np.array([q1, v1, q2, v2])
                    i += 1

    return state_array

# Function to create states array taken from a uniform distribution
def random_states(n_q1v1):
    state_array = np.zeros((n_q1v1 * 441, 4))

    for i in range(n_q1v1):
        # Generate a random pair of q1 and v1
        q1 = (upperPositionLimit_q1 - lowerPositionLimit_q1) * np.random.random_sample() + lowerPositionLimit_q1
        v1 = (upperVelocityLimit_v1 - lowerVelocityLimit_v1) * np.random.random_sample() + lowerVelocityLimit_v1

        for j in range(441):
            # Generate random pairs of q2 and v2 for the random pair of q1 and v1
            q2 = (upperPositionLimit_q2 - lowerPositionLimit_q2) * np.random.random_sample() + lowerPositionLimit_q2
            v2 = (upperVelocityLimit_v2 - lowerVelocityLimit_v2) * np.random.random_sample() + lowerVelocityLimit_v2
            state_array[i * 441 + j, :] = np.array([q1, v1, q2, v2])

    return state_array



