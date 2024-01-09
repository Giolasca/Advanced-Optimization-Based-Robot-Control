import numpy as np
import casadi
import matplotlib.pyplot as plt
import double_pendulum_dynamics as double_pendulum_dynamics
import mpc_double_pendulum_conf as conf
from orc.A3.DoublePendulum.neural_network import create_model

class MpcDoublePendulum:

    def __init__(self):
        self.T = conf.T                  # OCP horizon
        self.dt = conf.dt                # time step
        self.w_q1 = conf.w_q1              # Position weight link1
        self.w_u1 = conf.w_u1              # Input weight link1
        self.w_v1 = conf.w_v1              # Velocity weight link1
        self.w_q2 = conf.w_q2              # Position weight link2
        self.w_u2 = conf.w_u2              # Input weight link2
        self.w_v2 = conf.w_v2              # Velocity weight link2
        self.N = int(self.T/self.dt)        # I initalize the Opti helper from casadi
        self.model = create_model(4)     # Template of NN (Dimension of input = 4)
        self.model.load_weights("viable_nonviable_model.h5")
        self.weights = self.model.get_weights()
    
    def nn_to_casadi(self, params, x):
        out = np.array(x)
        it = 0

        for param in params:
            param = np.array(param.tolist())

            if it % 2 == 0:
                out = out @ param
            else:
                out = param + out
                for i, item in enumerate(out):
                    out[i] = casadi.fmax(0., casadi.MX(out[i]))

            it += 1
        
        return casadi.MX(out[0])

    def solve(self, x_init, X_guess = None, U_guess = None):
        self.opti = casadi.Opti()                   # N is the size of the vector we want to realize, the number of steps we want to compute. We create a vector containing all the states of size N+1, 
                                                    # We create a vector of control inputs of size N, one less than the list of states since final control input doesn't have any importance
        # Casadi variables declaration
        self.q1 = self.opti.variable(self.N+1)       
        self.v1 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        self.q2 = self.opti.variable(self.N+1)       
        self.v2 = self.opti.variable(self.N+1)
        self.u2 = self.opti.variable(self.N)
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1
        q2 = self.q2
        v2 = self.v2
        u2 = self.u2

        # State vector initialization
        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess[0,i])
                self.opti.set_initial(v1[i], X_guess[1,i])
                self.opti.set_initial(q2[i], X_guess[2,i])
                self.opti.set_initial(v2[i], X_guess[3,i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], x_init[0])
                self.opti.set_initial(v1[i], x_init[1])
                self.opti.set_initial(q2[i], x_init[2])
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
        q1_target = casadi.pi*5/4        # Target position of the first link  
        q2_target = casadi.pi*5/4        # Target position of the second link  


        # Cost definition
        i = 0
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
            x_next = double_pendulum_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
            # Dynamics imposition
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
            # Position bounds
            self.opti.subject_to(q1[i] <= conf.upperPositionLimit_q1)
            self.opti.subject_to(q1[i] >= conf.lowerPositionLimit_q1)
            self.opti.subject_to(q2[i] <= conf.upperPositionLimit_q2)
            self.opti.subject_to(q2[i] >= conf.lowerPositionLimit_q2)

            # Velocity bounds
            self.opti.subject_to(v1[i] <= conf.upperVelocityLimit_v1)
            self.opti.subject_to(v1[i] >= conf.lowerVelocityLimit_v1)
            self.opti.subject_to(v2[i] <= conf.upperVelocityLimit_v2)
            self.opti.subject_to(v2[i] >= conf.lowerVelocityLimit_v2)
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(u1[i] <= conf.upperControlBound_u1)
                self.opti.subject_to(u1[i] >= conf.lowerControlBound_u1)
                self.opti.subject_to(u2[i] <= conf.upperControlBound_u2)
                self.opti.subject_to(u2[i] >= conf.lowerControlBound_u2)

        # Terminal constraint (NN)
        state = [q1[self.N], v1[self.N], q2[self.N], v2[self.N]]
        self.opti.subject_to(self.nn_to_casadi(self.weights, state) > 0)

        return self.opti.solve()


if __name__ == "__main__":

    # Instance of MCP solver
    mpc = MpcDoublePendulum()
    
    initial_state = np.array([3/4*np.pi, 0, np.pi, 10])      # Initial state of the pendulum (position and velocity)
    actual_trajectory = []                  # Lists to store the trajectory during the MPC iterations
    actual_inputs = []                      # Lists to store the inputs during the MPC iterations

    n_step = 100        # Number of MPC steps
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    # First run
    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[0]), sol.value(mpc.v1[0]), sol.value(mpc.q2[0]), sol.value(mpc.v2[0])]))
    actual_inputs.append(sol.value(mpc.u1[0]), sol.value(mpc.u2[0]))

    # We need to create a state_guess of size 2 x N+1
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q1[i+1])
        new_state_guess[1, i] = sol.value(mpc.v1[i+1])
        new_state_guess[2, i] = sol.value(mpc.q2[i+1])
        new_state_guess[3, i] = sol.value(mpc.v2[i+1])

    # We need to create an input_guess of size N
    for i in range(mpc.N-1):
        new_input_guess[0, i] = sol.value(mpc.u2[i+1])
        new_input_guess[1, i] = sol.value(mpc.u2[i+1])
    
    # Update the state and input guesses for the next MPC iteration
    for i in range(n_step):
        sol = mpc.solve(actual_trajectory[i], new_state_guess, new_input_guess)
        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append(sol.value(mpc.u1[0]), sol.value(mpc.u2[0]))

        # Update state_guess for the next iteration
        for i in range(mpc.N):
            new_state_guess[0, i] = sol.value(mpc.q1[i+1])
            new_state_guess[1, i] = sol.value(mpc.v1[i+1])
            new_state_guess[2, i] = sol.value(mpc.q2[i+1])
            new_state_guess[3, i] = sol.value(mpc.v2[i+1])

        # Update input_guess for the next iteration
        for i in range(mpc.N-1):
            new_input_guess[0, i] = sol.value(mpc.u2[i+1])
            new_input_guess[1, i] = sol.value(mpc.u2[i+1])
        
    positions = []
    velocities = []

    # Extract positions and velocities from the actual trajectory
    for i, state in enumerate(actual_trajectory):
        positions.append(actual_trajectory[i][0])
        velocities.append(actual_trajectory[i][1])

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.scatter(positions, velocities, c='r')
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(actual_inputs)
    plt.title('Actual Inputs')
    plt.xlabel('Steps')
    plt.ylabel('Inputs')
    plt.show()