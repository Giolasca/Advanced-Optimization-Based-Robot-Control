import numpy as np

multiproc = 1
num_processes = 4

T = 1                   # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
lowerVelocityLimit = -10
upperVelocityLimit = 10
lowerControlBound = -9.81
upperControlBound = 9.81

w_q = 10
w_v = 10
w_u = 1e-2