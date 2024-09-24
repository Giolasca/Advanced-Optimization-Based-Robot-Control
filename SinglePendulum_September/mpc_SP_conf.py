import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

### Horizon parameters
TC = 1          # Terminal cost
costraint = 0   # handles constraints in the main code

if TC: 
    T = 0.01    # OCP horizon
    N = 6 
else:
    T = 1     # OCP horizon
    N = 50 

dt = 0.01        # OCP time step
#N = int(T/dt)    # Number of horizon step

# Maximum number of iterations for the solver
max_iter = 50

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
#w_tc = 1e2

# Number of MPC steps to simulate
mpc_step = 50

# Model file name
nn = "nn_SP_135_constr.h5"
#nn = "nn_SP_135_unconstr.h5"
#nn = "nn_SP_180_constr.h5"
#nn = "nn_SP_180_unconstr.h5"
#nn = "nn_SP_225_constr.h5"
#nn = "nn_SP_225_unconstr.h5"
#nn = "nn_SP_225_linear_unconstr.h5"

# Initial and Target state 
if nn in ["nn_SP_135_constr.h5", "nn_SP_135_unconstr.h5"]:
    initial_state = np.array([5/4*np.pi, 0])
    q_target = 3/4 * np.pi
elif nn in ["nn_SP_180_constr.h5", "nn_SP_180_unconstr.h5"]:
    initial_state = np.array([3/4*np.pi, 0])
    q_target = 4/4 * np.pi
elif nn in ["nn_SP_225_constr.h5", "nn_SP_225_unconstr.h5", "nn_SP_225_linear_unconstr.h5"]:
    initial_state = np.array([3/4*np.pi, 0])
    q_target = 5/4 * np.pi
else:
    q_target = None
    print("File name not recognized. q_target not set.")

# Function to load dataframe based on the nn filename
def load_dataframe(nn):
    if nn == "nn_SP_135_constr.h5":
        return pd.read_csv("ocp_data_SP_target_135_constr.csv")
    elif nn == "nn_SP_135_unconstr.h5":
        return pd.read_csv("ocp_data_SP_target_135_unconstr.csv")
    if nn == "nn_SP_180_constr.h5":
        return pd.read_csv("ocp_data_SP_target_180_constr.csv")
    elif nn == "nn_SP_180_unconstr.h5":
        return pd.read_csv("ocp_data_SP_target_180_unconstr.csv")
    if nn == "nn_SP_225_constr.h5":
        return pd.read_csv("ocp_data_SP_target_225_constr.csv")
    elif nn == "nn_SP_225_unconstr.h5":
        return pd.read_csv("ocp_data_SP_target_225_unconstr.csv")
    elif nn == "nn_SP_225_linear_unconstr.h5":
        return pd.read_csv("ocp_data_SP_target_225_linear_unconstr.csv")
    else:
        raise ValueError("Unknown nn filename")

# Load dataset
dataframe = load_dataframe(nn)
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

# Function to create states array in a grid
def grid_states(n_pos, n_vel):
    n_ics = n_pos * n_vel
    possible_q = np.linspace(q_min, q_max, num=n_pos)
    possible_v = np.linspace(v_min, v_max, num=n_vel)
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
        possible_q = (q_max - q_min) * np.random.random_sample() + q_min
        possible_v = (v_max - v_min) * np.random.random_sample() + v_min
        state_array[i,:] = np.array([possible_q, possible_v])
    
    return n_states, state_array