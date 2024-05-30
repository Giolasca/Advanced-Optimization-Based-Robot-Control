import numpy as np
import casadi
import SP_dynamics as F_single_pendulum_dynamics
import mpc_SP_conf as conf
from nn_single import create_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

class MpcSinglePendulum:

    def __init__(self):
        self.T = conf.T                     # MPC horizon
        self.dt = conf.dt                   # time step
        self.w_q = conf.w_q                 # Position weight
        self.w_u = conf.w_u                 # Input weight
        self.w_v = conf.w_v                 # Velocity weight
        self.N = int(self.T/self.dt)        # Number of steps
        self.model = create_model(2)        # Template of NN
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
        self.q = self.opti.variable(self.N+1)       
        self.v = self.opti.variable(self.N+1)
        self.u = self.opti.variable(self.N)
        q = self.q
        v = self.v
        u = self.u
        
        # State vector initialization
        if X_guess is not None:
            for i in range(self.N+1):
                self.opti.set_initial(q[i], X_guess[0, i])
                self.opti.set_initial(v[i], X_guess[1, i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q[i], x_init[0])
                self.opti.set_initial(v[i], x_init[1])
        
        # Control input vector initialization
        if U_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(u[i], U_guess[i])

        state = [(q[self.N] - self.mean[0])/self.std[0], (v[self.N] - self.mean[1])/self.std[1]]
        
        # Target position
        q_target = conf.q_target

        # Cost definition
        self.cost = 0
        self.running_costs = [None] * (self.N+1)
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            self.running_costs[i] += self.w_q * (q[i] - q_target)**2
            if i < self.N:
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        
        # Adding terminal cost from the neural network
        self.cost += self.nn_to_casadi(self.weights, state)
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            x_next = F_single_pendulum_dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])

        # Initial state constraint       
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # Bounds constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit, q[i], conf.upperPositionLimit))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit, v[i], conf.upperVelocityLimit))
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(conf.lowerControlBound, u[i], conf.upperControlBound))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)

        return self.opti.solve()


if __name__ == "__main__":
    # Instance of OCP solver
    mpc = MpcSinglePendulum()

    initial_state = conf.initial_state
    actual_trajectory = []
    actual_inputs = []

    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((2, mpc.N+1))
    new_input_guess = np.zeros((mpc.N))

    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
    actual_inputs.append(sol.value(mpc.u[0]))
    
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q[i+1])
        new_state_guess[1, i] = sol.value(mpc.v[i+1])
    for i in range(mpc.N-1):
        new_input_guess[i] = sol.value(mpc.u[i+1])
        
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

        terminal_cost_value = sol.value(mpc.nn_to_casadi(mpc.weights, [sol.value(mpc.q[mpc.N]), sol.value(mpc.v[mpc.N])]))

        actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
        actual_inputs.append(sol.value(mpc.u[0]))
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q[k+1])
            new_state_guess[1, k] = sol.value(mpc.v[k+1])
        for j in range(mpc.N-1):
            new_input_guess[j] = sol.value(mpc.u[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_costs[i]), "Terminal Cost", terminal_cost_value)
        
    positions = []
    velocities = []

    for i, state in enumerate(actual_trajectory):
        positions.append(actual_trajectory[i][0])
        velocities.append(actual_trajectory[i][1])

    positions = []
    velocities = []

    for element in actual_trajectory:
        positions.append(element[0])
        velocities.append(element[1])

    # Position plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(positions)
    plt.xlabel('mpc step')
    plt.ylabel('q [rad]')
    plt.title('Position')
    plt.show()

    # Velocity plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(velocities)
    plt.xlabel('mpc step')
    plt.ylabel('v [rad/s]')
    plt.title('Velocity')
    plt.show()

    # Torque plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(actual_inputs)
    plt.xlabel('mpc step')
    plt.ylabel('u [N/m]')
    plt.title('Torque')
    plt.show()

    # Create a DataFrame starting from the final array
    columns = ['Pos_q1']
    df = pd.DataFrame(positions, columns=columns)

