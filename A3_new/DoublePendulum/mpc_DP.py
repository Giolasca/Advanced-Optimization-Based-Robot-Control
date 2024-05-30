import numpy as np
import casadi
import DP_dynamics as F_double_pendulum_dynamics
import mpc_DP_conf as conf
from nn import create_model  # Assuming you have the NN setup for double pendulum
import matplotlib.pyplot as plt
import pandas as pd

class MpcDoublePendulum:

    def __init__(self):
        self.T = conf.T                     # MPC horizon
        self.dt = conf.dt                   # time step
        self.w_q = conf.w_q                 # Position weight
        self.w_u = conf.w_u                 # Input weight
        self.w_v = conf.w_v                 # Velocity weight
        self.N = int(self.T/self.dt)        # Number of steps
        self.model = create_model(4)        # Template of NN for double pendulum
        self.model.load_weights("ocp_nn_model.h5")
        self.weights = self.model.get_weights()
        self.mean, self.std = conf.init_scaler()
    
    def nn_to_casadi(self, params, x):
        out = np.array(x)
        iteration = 0

        for param in params:
            param = np.array(param.tolist())

            if iteration % 2 == 0:
                out = out @ param
            else:
                out = param + out
                for i, item in enumerate(out):
                    out[i] = casadi.fmax(0., casadi.MX(out[i]))

            iteration += 1

        return casadi.MX(out[0])
    
    def solve(self, x_init, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()  # Initialize the optimizer
        # Casadi variables declaration
        self.q1 = self.opti.variable(self.N+1)
        self.v1 = self.opti.variable(self.N+1)
        self.q2 = self.opti.variable(self.N+1)
        self.v2 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        self.u2 = self.opti.variable(self.N)
        q1, v1, q2, v2, u1, u2 = self.q1, self.v1, self.q2, self.v2, self.u1, self.u2
        
        # State vector initialization
        if X_guess is not None:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess[0, i])
                self.opti.set_initial(v1[i], X_guess[1, i])
                self.opti.set_initial(q2[i], X_guess[2, i])
                self.opti.set_initial(v2[i], X_guess[3, i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], x_init[0])
                self.opti.set_initial(v1[i], x_init[1])
                self.opti.set_initial(q2[i], x_init[2])
                self.opti.set_initial(v2[i], x_init[3])
        
        # Control input vector initialization
        if U_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess[0, i])
                self.opti.set_initial(u2[i], U_guess[1, i])

        state = [(q1[self.N] - self.mean[0])/self.std[0], (v1[self.N] - self.mean[1])/self.std[1], 
                 (q2[self.N] - self.mean[2])/self.std[2], (v2[self.N] - self.mean[3])/self.std[3]]
        
        # Target positions
        q1_target, q2_target = conf.q1_target, conf.q2_target

        # Cost definition
        self.cost = 0
        self.running_costs = [None] * (self.N+1)
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * (v1[i]**2 + v2[i]**2)
            self.running_costs[i] += self.w_q * ((q1[i] - q1_target)**2 + (q2[i] - q2_target)**2)
            if i < self.N:
                self.running_costs[i] += self.w_u * (u1[i]**2 + u2[i]**2)
            self.cost += self.running_costs[i]
        
        # Adding terminal cost from the neural network
        self.cost += self.nn_to_casadi(self.weights, state)
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            x_next = F_double_pendulum_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(v1[i+1] == x_next[1])
            self.opti.subject_to(q2[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])

        # Initial state constraint       
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])

        # Bounds constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit1, q1[i], conf.upperPositionLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit1, v1[i], conf.upperVelocityLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit2, q2[i], conf.upperPositionLimit2))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit2, v2[i], conf.upperVelocityLimit2))
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(conf.lowerControlBound1, u1[i], conf.upperControlBound1))
            self.opti.subject_to(self.opti.bounded(conf.lowerControlBound2, u2[i], conf.upperControlBound2))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)

        return self.opti.solve()


if __name__ == "__main__":
    # Instance of OCP solver
    mpc = MpcDoublePendulum()

    initial_state = conf.initial_state
    actual_trajectory = []
    actual_inputs = []

    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
    actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
    
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q1[i+1])
        new_state_guess[1, i] = sol.value(mpc.v1[i+1])
        new_state_guess[2, i] = sol.value(mpc.q2[i+1])
        new_state_guess[3, i] = sol.value(mpc.v2[i+1])
    for i in range(mpc.N-1):
        new_input_guess[0, i] = sol.value(mpc.u1[i+1])
        new_input_guess[1, i] = sol.value(mpc.u2[i+1])
        
    for i in range(mpc_step):
        init_state = actual_trajectory[i]
        try:
            sol = mpc.solve(init_state, new_state_guess, new_input_guess)
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("")
                print("======================================")
                print("MPC stopped due to infeasible problem")
                print("======================================")
                print("")
                print(mpc.opti.debug.show_infeasibilities())
                break
            else:
                print(e)

        terminal_cost_value = sol.value(mpc.nn_to_casadi(mpc.weights, [sol.value(mpc.q1[mpc.N]), sol.value(mpc.v1[mpc.N]), sol.value(mpc.q2[mpc.N]), sol.value(mpc.v2[mpc.N])]))

        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q1[k+1])
            new_state_guess[1, k] = sol.value(mpc.v1[k+1])
            new_state_guess[2, k] = sol.value(mpc.q2[k+1])
            new_state_guess[3, k] = sol.value(mpc.v2[k+1])
        for j in range(mpc.N-1):
            new_input_guess[0, j] = sol.value(mpc.u1[j+1])
            new_input_guess[1, j] = sol.value(mpc.u2[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_costs[i]), "Terminal Cost", terminal_cost_value)
        
    positions_q1 = []
    velocities_v1 = []
    positions_q2 = []
    velocities_v2 = []

    for i, state in enumerate(actual_trajectory):
        positions_q1.append(actual_trajectory[i][0])
        velocities_v1.append(actual_trajectory[i][1])
        positions_q2.append(actual_trajectory[i][2])
        velocities_v2.append(actual_trajectory[i][3])

    # Position plot for q1
    fig = plt.figure(figsize=(12,8))
    plt.plot(positions_q1)
    plt.xlabel('mpc step')
    plt.ylabel('q1 [rad]')
    plt.title('Position q1')
    plt.show()

    # Velocity plot for v1
    fig = plt.figure(figsize=(12,8))
    plt.plot(velocities_v1)
    plt.xlabel('mpc step')
    plt.ylabel('v1 [rad/s]')
    plt.title('Velocity v1')
    plt.show()

    # Position plot for q2
    fig = plt.figure(figsize=(12,8))
    plt.plot(positions_q2)
    plt.xlabel('mpc step')
    plt.ylabel('q2 [rad]')
    plt.title('Position q2')
    plt.show()

    # Velocity plot for v2
    fig = plt.figure(figsize=(12,8))
    plt.plot(velocities_v2)
    plt.xlabel('mpc step')
    plt.ylabel('v2 [rad/s]')
    plt.title('Velocity v2')
    plt.show()

    # Torque plot for u1 and u2
    fig = plt.figure(figsize=(12,8))
    plt.plot([u[0] for u in actual_inputs], label='u1')
    plt.plot([u[1] for u in actual_inputs], label='u2')
    plt.xlabel('mpc step')
    plt.ylabel('u [N/m]')
    plt.title('Torque')
    plt.legend()
    plt.show()
