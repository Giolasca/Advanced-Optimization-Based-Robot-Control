import numpy as np
import casadi
import single_pendulum_dynamics as single_pendulum_dynamics
import mpc_single_pendulum_conf as conf
from neural_network import create_model

class MpcSinglePendulum:

    def __init__(self):
        self.T = conf.T                     # MPC horizon
        self.dt = conf.dt                   # time step
        self.w_q = conf.w_q                 # Position weight
        self.w_u = conf.w_u                 # Input weight
        self.w_v = conf.w_v                 # Velocity weight
        self.N = int(self.T/self.dt)        # I initalize the Opti helper from casadi
        self.model = create_model(2)        # Template of NN
        self.model.load_weights("viable_nonviable_model.h5")
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
        self.q = self.opti.variable(self.N+1)       
        self.v = self.opti.variable(self.N+1)
        self.u = self.opti.variable(self.N)
        q = self.q
        v = self.v
        u = self.u
        
        # State vector initialization
        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(q[i], X_guess[0,i])
                self.opti.set_initial(v[i], X_guess[1,i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q[i], x_init[0])
                self.opti.set_initial(v[i], x_init[1])
        
        # Control input vector initalization
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u[i], U_guess[i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Target position
        q_target = casadi.pi*7/8

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            self.running_costs[i] += self.w_q * (q[i] - q_target)**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = single_pendulum_dynamics.f(np.array([q[i], v[i]]), u[i])
            # Dynamics imposition
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])
        
        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit, q[i], conf.upperPositionLimit))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit, v[i], conf.upperVelocityLimit))
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound, u[i], conf.upperControlBound))
        
        # Terminal constraint (NN)
        state = [q[self.N], v[self.N]]
        if conf.terminal_constraint_on:
            self.opti.subject_to(self.nn_to_casadi(self.weights, state) > 0.5)
        
        self.opti.callback(lambda i: print(i))

        return self.opti.solve()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
    # Dobbiamo creare una state_guess 2 x N+1
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q[i+1])
        new_state_guess[1, i] = sol.value(mpc.v[i+1])
    # Dobbiamo creare una input_guess di dimensione N
    for i in range(mpc.N-1):
        new_input_guess[i] = sol.value(mpc.u[i+1])
        
    for i in range(mpc_step):
        noise = np.random.normal(conf.mean, conf.std, actual_trajectory[i].shape)
        if conf.noise:
            init_state = actual_trajectory[i] + noise
        else:
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
        actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
        actual_inputs.append(sol.value(mpc.u[0]))
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q[k+1])
            new_state_guess[1, k] = sol.value(mpc.v[k+1])
        for j in range(mpc.N-1):
            new_input_guess[j] = sol.value(mpc.u[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        
    positions = []
    velocities = []

    for i, state in enumerate(actual_trajectory):
        positions.append(actual_trajectory[i][0])
        velocities.append(actual_trajectory[i][1])

    _, state_array = conf.grid_states(51,51)
    label_pred = mpc.model.predict(state_array)

    viable_states = []
    no_viable_states = []

    for i, label in enumerate(label_pred):
        if label>0.5:
            viable_states.append(state_array[i,:])
        else:
            no_viable_states.append(state_array[i,:])
    
    viable_states = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.scatter(positions, velocities, c='g')
    ax.scatter(viable_states[:,0], viable_states[:,1], c='r')
    ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b')
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

    # Torque plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(actual_inputs)
    plt.show()