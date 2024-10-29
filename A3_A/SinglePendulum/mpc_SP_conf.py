import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


#==================================================================#
#                     Parameters to change
#==================================================================#

### Horizon parameters
TC = 1          # Terminal cost
dt = 0.01       # OCP time step
costraint = 0   # handles constraints in the main code
scenario_type = 1  # Introduce scenario type: 1 for T = 1, 0 for T = 0.01

# Noise parameters for robustness testing
noise = 1           # Test also robusteness to external disturbances
mean = 0.001        # Noise mean 
std = 0.001         # Noise std

# multiple test with different initial states
multi_test = 1
manual = 1

# Model file name for the neural network
#nn = "nn_SP_135_unconstr.h5"
#nn = "nn_SP_180_unconstr.h5"
nn = "nn_SP_225_unconstr_tanh.h5"


# Set T based on both TC and scenario_type
if TC:
    T = 0.01  # If terminal cost, T is fixed at 0.01
    N = int(T/dt)

if (TC == 0 and scenario_type == 1):
    T = 1   # Scenario with T = 1
    N = int(T/dt)

if (TC == 0 and scenario_type == 0):
    T = 0.01  # Scenario with T = 0.01 
    N = int(T/dt)



#==================================================================#
#                     Parameters of the MPC
#==================================================================#

# Maximum number of iterations for the solver
max_iter = 50

# Number of MPC steps to simulate
mpc_step = 20

### Constaints for the pendulum ###
q_min = 3/4*np.pi
q_max = 5/4*np.pi
v_min = -10
v_max = 10
u_min = -9.81
u_max = 9.81

### Weights for the pendulum ###
w_q = 1e2       # Weight for position
w_v = 1e-1      # Weight for velocity
w_u = 1e-4      # Weight for input



#==================================================================#
#                 Load and set data for simulation
#==================================================================#

# Perform Multi Tests with Different Initial States
if multi_test == 1:
    num_tests = 15
    if manual == 1:
        # List of manually defined initial values
        initial_values = [
            [3.7478, -0.7237],
            [2.3561,  0.1456],
            [3.0121, -2.0124],
            [3.8421, -5.3423],
            [3.6754, -2.4532],
            [3.3463,  2.1887],
            [3.6225,  0.9876],
            [2.3781,  1.9873],
            [2.9284,  2.3488],
            [3.2145, -2.9784],
            [3.5622,  3.1844],
            [3.4671, -0.9276],
            [3.2843,  4.9733],
            [3.3843,  4.2953],
            [3.6714, -1.2197],
        ]
        initial_states = np.array(initial_values)  # Convert the list to a NumPy array
    else: 
        # Generate random initial states if manual mode is not selected
        q_range = [3/4*np.pi, 5/4*np.pi]
        v_range = [-10, 10]
        
        initial_states = np.zeros((num_tests, 2))  
        initial_states[:, 0] = np.random.uniform(q_range[0], q_range[1], num_tests)
        initial_states[:, 1] = np.random.uniform(v_range[0], v_range[1], num_tests)

else:
    num_tests = 1  # Set num_tests to 1 if multi_test is not activated


# Select neural network and target 
if nn == "nn_SP_135_unconstr.h5":
    if multi_test == 0:
        initial_states = np.array([[5/4 * np.pi, 0]])
    q_target = 3/4 * np.pi
elif nn == "nn_SP_180_unconstr.h5":
    if multi_test == 0:
        initial_states = np.array([[3/4 * np.pi, 0]])
    q_target = 4/4 * np.pi
elif nn == "nn_SP_225_unconstr_tanh.h5":
    if multi_test == 0:
        initial_states = np.array([[3/4 * np.pi, 0]])
    q_target = 5/4 * np.pi
else:
    q_target = None
    print("File name not recognized. q_target not set.")

# Function to load dataframe based on the nn filename
def load_dataframe(nn):
    if nn == "nn_SP_135_unconstr.h5":
        return pd.read_csv("ocp_data_SP_target_135_constr.csv")
    elif nn == "nn_SP_180_unconstr.h5":
        return pd.read_csv("ocp_data_SP_target_180_unconstr.csv")
    elif nn == "nn_SP_225_unconstr_tanh.h5":
        return pd.read_csv("ocp_data_SP_target_225_unconstr.csv")
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