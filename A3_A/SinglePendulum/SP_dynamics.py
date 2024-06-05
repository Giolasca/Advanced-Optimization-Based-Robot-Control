from casadi import sin 
import numpy as np 

def f(x,u):
    dt = 0.1
    l = 1
    m = 1
    g = 9.81

    q, dq = x[0], x[1]

    ddq = -g/l*sin(q) + 1/(m * l**2)*u

    x_next = x + dt*np.array([dq,ddq])
    
    return x_next