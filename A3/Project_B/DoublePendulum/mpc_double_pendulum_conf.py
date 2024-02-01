import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

T = 0.1          # OCP horizion
dt = 0.01        # OCP time step
max_iter = 100     # Maximum iteration per point

terminal_constraint_on = 1
initial_state = np.array([3/4*np.pi, 0, 3/4*np.pi, 5])
q1_target = 4/4*np.pi
q2_target = 4/4*np.pi
noise = 0
mean = 0
std = 0.1

mpc_step = 200

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


###  Dataset  ###
dataframe = pd.read_csv("data_double_171780.csv")
labels = dataframe['viable']
dataset = dataframe.drop('viable', axis=1)
train_size = 0.8
scaler = StandardScaler()
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=46)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std


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