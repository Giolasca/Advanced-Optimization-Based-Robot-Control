import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Flag for imposing terminal cost
'''
terminal_cost = 0
if terminal_cost:
    T = 0.5                 # MPC horizion
else:
    T = 1
'''
T = 0.5
terminal_cost = 0

dt = 0.01               # MPC time step
max_iter = 100          # Maximum iteration per point

initial_state = np.array([np.pi, 0])
q_target = 5/4*np.pi

mpc_step = 30

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

###  Dataset  ###
dataframe = pd.read_csv("ocp_data.csv")
labels = dataframe['cost']
dataset = dataframe.drop('cost', axis=1)
train_size = 0.8
scaler = StandardScaler()
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=29)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std
