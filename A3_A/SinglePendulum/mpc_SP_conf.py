import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

### Horizon parameters
TC = 1           # Terminal cost

if TC: 
    T = 0.5      # OCP horizon 
else:
    T = 1.0      # OCP horizon 

dt = 0.01        # OCP time step
N = int(T/dt)    # Number of horizon step


# Maximum number of iterations for the solver
max_iter = 100   

### Constaints for the pendulum ###
q_min = 3/4*np.pi
q_max = 5/4*np.pi
v_min = -10
v_max = 10
u_min = -9.81
u_max = 9.81

### Weights for the pendulum ###
w_q = 1e2       # Weight for position
w_v = 1e-1      # Weight for input
w_u = 1e-4      # Weight for velocity

### Initial and target state
initial_state = np.array([3/4*np.pi, 0])
q_target = 5/4 * np.pi

# Number of MPC steps to simulate
mpc_step = 200

### Constaints for the pendulum ###
lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
lowerVelocityLimit = -10
upperVelocityLimit = 10
lowerControlBound = -9.81
upperControlBound = 9.81

###  Dataset  
dataframe = pd.read_csv("ocp_data_SP.csv")
labels = dataframe['Costs']
dataset = dataframe.drop('Costs', axis=1)
train_size = 0.8
scaler = StandardScaler()
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=29)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std


# Function to create states array in a grid
def grid_states(n_pos, n_vel):
    n_ics = n_pos * n_vel
    possible_q = np.linspace(lowerPositionLimit, upperPositionLimit, num=n_pos)
    possible_v = np.linspace(lowerVelocityLimit, upperVelocityLimit, num=n_vel)
    state_array = np.zeros((n_ics, 2))

    j = k = 0
    for i in range (n_ics):
        state_array[i,:] = np.array([possible_q[j], possible_v[k]])
        k += 1
        if (k == n_vel):
            k = 0
            j += 1

    return n_ics, state_array


# Function to create states array taken from a uniform distribution
def random_states(n_states):
    state_array = np.zeros((n_states, 2))

    for i in range(n_states):
        possible_q = (upperPositionLimit - lowerPositionLimit) * np.random.random_sample() + lowerPositionLimit
        possible_v = (upperVelocityLimit - lowerVelocityLimit) * np.random.random_sample() + lowerVelocityLimit
        state_array[i,:] = np.array([possible_q, possible_v])
    
    return n_states, state_array