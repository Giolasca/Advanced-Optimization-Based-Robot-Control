import numpy as np

multiproc = 1
num_processes = 4

T = 1                   # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

lowerPositionLimit1 = 3/4*np.pi
upperPositionLimit1 = 5/4*np.pi
lowerVelocityLimit1 = -10
upperVelocityLimit1 = 10

lowerControlBound1 = -9.81
upperControlBound1 = 9.81

lowerPositionLimit2 = 3/4*np.pi
upperPositionLimit2 = 5/4*np.pi
lowerVelocityLimit2 = -10
upperVelocityLimit2 = 10

lowerControlBound2 = -9.81
upperControlBound2 = 9.81

w_q1 = 10
w_v1 = 10
w_u1 = 1e-2

w_q2 = 10
w_v2 = 10
w_u2= 1e-2