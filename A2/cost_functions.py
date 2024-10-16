# -*- coding: utf-8 -*-
import numpy as np

class Empty:
    def __init__(self):
        pass

class OCPFinalCostState:
    ''' Cost function for reaching a desired state of the robot '''
    def __init__(self, name, v_des, weight_vel):
        self.name = name
        self.nq = 1
        self.v_des = v_des
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        v = x[self.nq:]
        de = v - self.v_des                             # TODO implement penalty on the final velocity
        cost = 0.5*self.weight_vel*np.dot(de, de)       # Adding a quadratic penalty on the velocity to the cost
        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        v = x[self.nq:]
        de = v - self.v_des                                 # TODO implement penalty on the final velocity
        cost = 0.5*self.weight_vel*np.dot(de, de)           # Adding a quadratic penalty on the velocity to the cost
        grad =  np.array([0, self.weight_vel*de[0]])          # TODO implement the gradient of the penalty on the final velocity
        return (cost, grad)
        
class OCPRunningCostQuadraticControl:
    ''' Quadratic cost function for penalizing control inputs '''
    def __init__(self, name, dt):
        self.name = name
        self.dt = dt
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        cost = 0.5*np.dot(u,u)        # TODO implement control regularization

        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        cost = 0.5*np.dot(u,u)                  # TODO implement control regularization
        grad_x = np.zeros(x.shape[0])           # TODO implement the gradient w.r.t. x of the control regularization
        grad_u = u                              # TODO implement the gradient w.r.t. u of the control regularization

        return (cost, grad_x, grad_u)