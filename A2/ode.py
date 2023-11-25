# -*- coding: utf-8 -*-
import numpy as np

class ODE:
    def __init__(self, name):
        self.name = name
        self.nu = 1
        
    def f(self, x, u, t):
        return np.zeros(x.shape)
             
class ODEPendulum(ODE):
    def __init__(self, name=''):
        ODE.__init__(self, name) 
        self.g = -9.81
        
    def f(self, x, u, t, jacobian=False):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = self.g*np.sin(x[0]) + u

        if(jacobian):
            # Numerical differentiation for Jacobian with respect to x
            epsilon_x = 1e-7
            df_dx = np.zeros((2, 2))
            for i in range(2):
                x_perturbed = x.copy()
                x_perturbed[i] += epsilon_x
                dx_perturbed = self.f(x_perturbed, u, t)
                df_dx[:, i] = (dx_perturbed - dx) / epsilon_x

            # Numerical differentiation for Jacobian with respect to u
            epsilon_u = 1e-7
            df_du = np.zeros((2, 1))
            u_perturbed = u + epsilon_u
            dx_perturbed = self.f(x, u_perturbed, t)
            df_du[:, 0] = (dx_perturbed - dx) / epsilon_u

            return dx, df_dx, df_du   
        return dx
# Test
ode_pendulum = ODEPendulum()
x = np.array([0.0, 1.0])  # Initial State
u = 2.0  # Control Input
t = 0.0  # Time

<<<<<<< Updated upstream
        return dx
=======
result = ode_pendulum.f(x, u, t, jacobian=True)
dx, df_dx, df_du = result

print("dx:", dx)
print("Jacobian w.r.t x (df_dx): \n", df_dx)
print("Jacobian w.r.t u (df_du): \n", df_du)
>>>>>>> Stashed changes
