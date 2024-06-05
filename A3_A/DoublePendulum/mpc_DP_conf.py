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

### Constaints for the first pendulum ###
q1_min = 3/4*np.pi
q1_max = 5/4*np.pi
v1_min = -10
v1_max = 10
u1_min = -9.81
u1_max = 9.81

### Constaints for the second pendulum ###
q2_min = 3/4*np.pi
q2_max = 5/4*np.pi
v2_min = -10
v2_max = 10
u2_min = -9.81
u2_max = 9.81

### Weights for the pendulum ###
w_q = 1e2       # Weight for position
w_v = 1e-1      # Weight for input
w_u = 1e-3      # Weight for velocity
w_tc = 1e3      # Weight for terminal cost

### Initial and target state
initial_state = np.array([3/4*np.pi, 0, 3/4*np.pi, 2])

# Target positions for the double pendulum
q1_target = 4/4*np.pi  # Target position for the first pendulum
q2_target = 4/4*np.pi  # Target position for the second pendulum

# Number of MPC steps to simulate
mpc_step = 100


###  Dataset  ###
dataframe = pd.read_csv("ocp_data_DP.csv")
labels = dataframe['Costs']
dataset = dataframe.drop('Costs', axis=1)
train_size = 0.95
scaler = StandardScaler()
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=29)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Initialize the scaler with mean and standard deviation for state normalization
def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std