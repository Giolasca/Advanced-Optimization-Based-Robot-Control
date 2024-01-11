from casadi import sin 
import numpy as np 

def f(x, u):
    #data definitions
    dt = 0.01           #time step
    l = 1               #length of the link
    m = 1               # mass of the link
    g = 9.81            # gravity

    # state extraction from parameters
    q, dq = x[0], x[1]

    ddq = -g/l*sin(q) + 1/(m * l**2)*u

    x_next = x + dt*np.array([dq,ddq])

    return x_next