import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

T = 0.1                  # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

terminal_constraint_on = 0
initial_state = np.array([np.pi, np.pi, 0, 0])
q1_target = 3/4 * np.pi
q2_target = 5/4 * np.pi

mpc_step = 100

noise = 0
mean = 0
std = 0.1

lowerPositionLimit1 = 3/4*np.pi
upperPositionLimit1 = 5/4*np.pi
lowerVelocityLimit1 = -10
upperVelocityLimit1 = 10
lowerControlBound1 = -9.81
upperControlBound1 = 9.81

lowerPositionLimit2 = 3/4*np.pi
upperPositionLimit2 = 5/4*np.pi
lowerVelocityLimit2 = -10
upperVelocityLimit2 = 10
lowerControlBound2 = -9.81
upperControlBound2 = 9.81

w_q1 = 1e2
w_v1 = 1e-1
w_u1 = 1e-4

w_q2 = 1e2
w_v2 = 1e-1
w_u2 = 1e-4

scaler = StandardScaler()

dataframe = pd.read_csv('double_data.csv')
labels = dataframe['viable_states']
dataset = dataframe.drop('viable_states', axis=1)
train_size = 0.8
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=17)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std

# Function to create states array in a grid
def grid_states(n_pos, n_vel):
    n_ics = n_pos * n_vel * n_pos * n_vel
    possible_q1 = np.linspace(lowerPositionLimit1, upperPositionLimit1, num=n_pos)
    possible_v1 = np.linspace(lowerVelocityLimit1, upperVelocityLimit1, num=n_vel)
    possible_q2 = np.linspace(lowerPositionLimit2, upperPositionLimit2, num=n_pos)
    possible_v2 = np.linspace(lowerVelocityLimit2, upperVelocityLimit2, num=n_vel)
    state_array = np.zeros((n_ics, 4))

    j = k = l = m = 0
    for i in range (n_ics):
        state_array[i,:] = np.array([possible_q1[j], possible_v1[k], possible_q2[l], possible_v2[m]])
        m += 1
        if (m == n_vel):
            m = 0
            l += 1
            if (l == n_pos):
                l = 0
                k += 1
                if (k == n_vel):
                    k = 0
                    j += 1

    return n_ics, state_array

# Function to create states array taken from a uniform distribution
def random_states(n_states):
    state_array = np.zeros((n_states, 4))

    for i in range(n_states):
        possible_q1 = (upperPositionLimit1 - lowerPositionLimit1) * np.random.random_sample() + lowerPositionLimit1
        possible_v1 = (upperVelocityLimit1 - lowerVelocityLimit1) * np.random.random_sample() + lowerVelocityLimit1
        possible_q2 = (upperPositionLimit2 - lowerPositionLimit2) * np.random.random_sample() + lowerPositionLimit2
        possible_v2 = (upperVelocityLimit2 - lowerVelocityLimit2) * np.random.random_sample() + lowerVelocityLimit2
        state_array[i,:] = np.array([possible_q1, possible_v1, possible_q2, possible_v2])
    
    return n_states, state_array