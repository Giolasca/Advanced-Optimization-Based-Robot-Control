import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

### Horizon parameters
TC = 1          # Terminal cost
costraint = 0   # handles constraints in the main code
scenario_type = 1  # Introduce scenario type: 1 for T = 1, 0 for T = 0.01

# Set T based on both TC and scenario_type
if TC:
    T = 0.01  # If terminal cost, T is fixed at 0.01
    N = 5
else:
    if scenario_type == 1:
        T = 1   # Scenario with T = 1
        N = 50
    else:
        T = 0.01  # Scenario with T = 0.01 
        N = 50

dt = 0.01        # OCP time step
#N = int(T/dt)    # Number of horizon step

# Maximum number of iterations for the solver
max_iter = 50

### Constaints for the fisrt link ###
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

# Number of MPC steps to simulate
mpc_step = 50

# Model file name
nn = "nn_DP_180_180_unconstr.h5"

# Initial and Target state 
q1_target = np.pi
q2_target = np.pi

initial_state = np.array([3/4*np.pi, 0, 3/4*np.pi, 2])

# Load dataframe
dataframe = pd.read_csv("combined_data_180_180.csv")
labels = dataframe['cost']
dataset = dataframe.drop('cost', axis=1)
train_size = 0.8

# Split data
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=29)

# Scale the input features
scaler_X = StandardScaler()
train_data = scaler_X.fit_transform(train_data)
test_data = scaler_X.transform(test_data)

# Fit scaler for labels
scaler_y = StandardScaler()
train_label = train_label.values.reshape(-1, 1)  # Reshape to fit scaler
test_label = test_label.values.reshape(-1, 1)
train_label = scaler_y.fit_transform(train_label)
test_label = scaler_y.transform(test_label)

# Function to initialize and return scalers' means and stds
def init_scaler():
    scaler_mean_X = scaler_X.mean_
    scaler_std_X = scaler_X.scale_
    scaler_mean_y = scaler_y.mean_
    scaler_std_y = scaler_y.scale_
    return scaler_mean_X, scaler_std_X, scaler_mean_y, scaler_std_y