import numpy as np

multiproc = 1
num_processes = 4
plot = 1
old = 1

T = 1          # OCP horizion
dt = 0.01        # OCP time step
max_iter = 100     # Maximum iteration per point

### Constaints for the first link ###
lowerPositionLimit_q1 = 3/4*np.pi
upperPositionLimit_q1 = 5/4*np.pi
lowerVelocityLimit_v1 = -10
upperVelocityLimit_v1 = 10
lowerControlBound_u1 = -9.81
upperControlBound_u1 = 9.81

### Weights for the first link ###
w_q1 = 10
w_v1 = 10
w_u1 = 1e-2


### Constaints for the second link ###
lowerPositionLimit_q2 = 3/4*np.pi
upperPositionLimit_q2 = 5/4*np.pi
lowerVelocityLimit_v2 = -10
upperVelocityLimit_v2 = 10
lowerControlBound_u2 = -9.81
upperControlBound_u2 = 9.81

### Weights for the second link ###
w_q2 = 10
w_v2 = 10
w_u2 = 1e-2