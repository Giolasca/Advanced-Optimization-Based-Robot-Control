import numpy as np
import casadi
import doublependulum_dynamics as double_pendulum_dynamics
import mpc_double_pendulum_conf as conf
from neural_network_double import create_model

class MpcDoublePendulum:

    def __init__(self):
        self.T = conf.T                             # MPC horizon
        self.dt = conf.dt                           # time step
        self.w_q1 = conf.w_q1                       # 1 Position weight
        self.w_u1 = conf.w_u1                       # 1 Input weight
        self.w_v1 = conf.w_v1                       # 1 Velocity weight
        self.w_q2 = conf.w_q2                       # 2 Position weight
        self.w_u2 = conf.w_u2                       # 2 Input weight
        self.w_v2 = conf.w_v2                       # 2 Velocity weight
        self.N = int(self.T/self.dt)                # I initalize the Opti helper from casadi
        self.model = create_model(4)                # Template of NN
        self.model.load_weights("../nn/double_pendulum.h5")
        self.weights = self.model.get_weights()
    
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
        
    def solve(self, x_init, X_guess = None, U_guess = None):
        self.opti = casadi.Opti()                   # N is the size of the vector we want to realize, the number of steps we want to compute. We create a vector containing all the states of size N+1, 
                                                    # We create a vector of control inputs of size N, one less than the list of states since final control input doesn't have any importance
        # Casadi variables declaration
        self.q1 = self.opti.variable(self.N+1)       
        self.v1 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1
        self.q2 = self.opti.variable(self.N+1)       
        self.v2 = self.opti.variable(self.N+1)
        self.u2 = self.opti.variable(self.N)
        q2 = self.q2
        v2 = self.v2
        u2 = self.u2
        
        # State vector initialization
        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess[0,i])
                self.opti.set_initial(q2[i], X_guess[1,i])
                self.opti.set_initial(v1[i], X_guess[2,i])
                self.opti.set_initial(v2[i], X_guess[3,i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], x_init[0])
                self.opti.set_initial(q2[i], x_init[1])
                self.opti.set_initial(v1[i], x_init[2])
                self.opti.set_initial(v2[i], x_init[3])
        
        # Control input vector initalization
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess[0,i])
                self.opti.set_initial(u2[i], U_guess[1,i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Target position
        q1_target = casadi.pi*3/4
        q2_target = casadi.pi*3/4

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            self.running_costs[i] += self.w_q1 * (q1_target - q1[i])**2 + self.w_q2 * (q2_target - q2[i])**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u1 * u1[i]**2 + self.w_u2 * u2[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = double_pendulum_dynamics.f(np.array([q1[i], q2[i], v1[i], v2[i]]), np.array([u1[i], u2[i]]))
            # Dynamics imposition
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(q2[i+1] == x_next[1])
            self.opti.subject_to(v1[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])
        
        # Initial state constraint
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(q2[0] == x_init[1])
        self.opti.subject_to(v1[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit1, q1[i], conf.upperPositionLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit2, q2[i], conf.upperPositionLimit2))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit1, v1[i], conf.upperVelocityLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit2, v2[i], conf.upperVelocityLimit2))
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound1, u1[i], conf.upperControlBound1))
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound2, u2[i], conf.upperControlBound2))
        
        # Terminal constraint (NN)
        state = [q1[self.N], v1[self.N], q2[self.N], v2[self.N]]
        self.opti.subject_to(self.nn_to_casadi(self.weights, state) > 0)

        return self.opti.solve()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mpc = MpcDoublePendulum()

    initial_state = conf.initial_state
    actual_trajectory = []
    actual_inputs = []

    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    # First run
    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v1[1]), sol.value(mpc.v2[1])]))
    actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q1[i+1])
        new_state_guess[1, i] = sol.value(mpc.q2[i+1])
        new_state_guess[2, i] = sol.value(mpc.v1[i+1])
        new_state_guess[3, i] = sol.value(mpc.v2[i+1])
    for i in range(mpc.N-1):
        new_input_guess[0, i] = sol.value(mpc.u1[i+1])
        new_input_guess[1, i] = sol.value(mpc.u2[i+1])
    
    for i in range(mpc_step):
        sol = mpc.solve(actual_trajectory[i], new_state_guess, new_input_guess)
        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v1[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q1[k+1])
            new_state_guess[1, k] = sol.value(mpc.q2[k+1])
            new_state_guess[2, k] = sol.value(mpc.v1[k+1])
            new_state_guess[3, k] = sol.value(mpc.v2[k+1])
        for j in range(mpc.N-1):
            new_input_guess[0, j] = sol.value(mpc.u1[j+1])
            new_input_guess[1, j] = sol.value(mpc.u2[j+1])
        print("Step", i+1, "out of", mpc_step, "done")

    ## PLOTTA
    q1 = []
    q2 = []

    for i, state in enumerate(actual_trajectory):
        q1.append(actual_trajectory[i][0])
        q2.append(actual_trajectory[i][1])

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.scatter(q1, q2, c='r')
    ax.legend()
    ax.set_xlabel('q1 [rad]')
    ax.set_ylabel('q2 [rad]')
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(actual_inputs)
    plt.show()