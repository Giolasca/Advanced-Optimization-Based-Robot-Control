import numpy as np

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
w_v = 1e-2
w_u = 1e-2