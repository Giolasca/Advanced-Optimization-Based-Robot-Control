# mpc_DP_conf.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Horizon length
T = 0.5  

# Time step
dt = 0.01  

### Weights for the first link ###
w_q1 = 1e2
w_v1 = 1e-1
w_u1 = 1e-4

### Weights for the second link ###
w_q2 = 1e2
w_v2 = 1e-1
w_u2 = 1e-4

# Target positions for the double pendulum
q1_target = 3/4*np.pi  # Target position for the first pendulum
q2_target = 5/4*np.pi  # Target position for the second pendulum

# Position limits
lowerPositionLimit1 = 3/4*np.pi
upperPositionLimit1 = 5/4*np.pi
lowerPositionLimit2 = 3/4*np.pi
upperPositionLimit2 = 5/4*np.pi


# Velocity limits
lowerVelocityLimit1 = -10.0
upperVelocityLimit1 = 10.0
lowerVelocityLimit2 = -10.0
upperVelocityLimit2 = 10.0

# Control input bounds
lowerControlBound1 = -9.81
upperControlBound1 = 9.81
lowerControlBound2 = -9.81
upperControlBound2 = 9.81

# Initial state [q1, v1, q2, v2]
initial_state = np.array([3/4*np.pi, 0, 3/4*np.pi, 5])

# Number of MPC steps to simulate
mpc_step = 100

# Maximum number of iterations for the solver
max_iter = 100


###  Dataset  ###
dataframe = pd.read_csv("combined_data.csv")
labels = dataframe['Costs']
dataset = dataframe.drop('Costs', axis=1)
train_size = 0.8
scaler = StandardScaler()
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=29)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Initialize the scaler with mean and standard deviation for state normalization
def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std
