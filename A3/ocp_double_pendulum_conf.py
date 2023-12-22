import numpy as np

multiproc = 1
num_processes = 3

T = 1                   # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

lowerPositionLimit = np.array([3/4*np.pi, 3/4*np.pi])
upperPositionLimit = np.array([5/4*np.pi, 5/4*np.pi])
lowerVelocityLimit = np.array([-10, -10])
upperVelocityLimit = np.array([10, 10])

lowerControlBound = np.array([-9.81, -9.81])
upperControlBound = np.array([9.81, 9.81])
w_q = np.array([10, 10])
w_v = np.array([10, 10])
w_u = 1e-2