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
nn = "nn_DP_180_180_unconstr.h5"

# Set T based on both TC and scenario_type
if TC:
    T = 0.05  # If terminal cost, T is fixed at 0.01
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
mpc_step = 40

### Constaints for the fisrt link ###
q1_min = 3/4*np.pi
q1_max = 5/4*np.pi
v1_min = -10
v1_max = 10
u1_min = -9.81
u1_max = 9.81

### WWeights for the first link ###
w_q1 = 1e2
w_v1 = 1e-1
w_u1 = 1e-4

### Constaints for the second link ###
q2_min = 3/4*np.pi
q2_max = 5/4*np.pi
v2_min = -10
v2_max = 10
u2_min = -9.81
u2_max = 9.81

### Weights for the second link###
w_q2 = 1e2
w_v2 = 1e-1
w_u2 = 1e-4



#==================================================================#
#                 Load and set data for simulation
#==================================================================#

# Perform Multi Tests with Different Initial States
if multi_test == 1:
    num_tests = 15
    if manual == 1:
        # List of manually defined initial values
        initial_values = [
            [3.4783,  9.3095, 3.3732,  6.0186],
            [2.3724, -9.6972, 3.5088, -8.6178],
            [2.9832,  1.1900, 3.2363,  0.7995],
            [2.5939, -1.1888, 3.7184,  3.3731],
            [3.7255,  1.1769, 2.4238,  0.6009],
            [2.5693,  3.5629, 3.2501,  1.3601],
            [3.0524, -2.1130, 3.8793, -3.1467],
            [3.0279,  9.5295, 2.9746, -1.4757],
            [2.4943, -2.0657, 2.3757, -5.6091],
            [2.4210, -3.9519, 3.9223, -9.3334],
            [2.4356,  8.7363, 3.0880,  9.7961],
            [3.0039,  2.3439, 2.5103,  9.2105],
            [2.4170, -6.2775, 3.5059,  1.5050],
            [3.8200, -1.9316, 3.1089,  7.8279],
            [3.6122, -5.6011, 3.3698, -7.0064],
        ]
        initial_states = np.array(initial_values)  # Convert the list to a NumPy array
    else: 
        q1_range = [3/4*np.pi, 5/4*np.pi]  
        v1_range = [-10, 10]  
        q2_range = [3/4*np.pi, 5/4*np.pi]  
        v2_range = [-10, 10]  

        # Initialize array for initial states (num_tests, 4)
        initial_states = np.zeros((num_tests, 4))  
        initial_states[:, 0] = np.random.uniform(q1_range[0], q1_range[1], num_tests)  # q1
        initial_states[:, 1] = np.random.uniform(v1_range[0], v1_range[1], num_tests)  # v1
        initial_states[:, 2] = np.random.uniform(q2_range[0], q2_range[1], num_tests)  # q2
        initial_states[:, 3] = np.random.uniform(v2_range[0], v2_range[1], num_tests)  # v2

else:
    num_tests = 1  # Set num_tests to 1 if multi_test is not activated

# Initial and Target state 
q1_target = np.pi
q2_target = np.pi

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