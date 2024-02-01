import numpy as np
import casadi
import double_pendulum_dynamics as double_pendulum_dynamics
import mpc_double_pendulum_conf as conf
from neural_network import create_model
import matplotlib.pyplot as plt
import pandas as pd

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
        self.N = int(self.T/self.dt)       # I initalize the Opti helper from casadi
        self.model = create_model(4)       # Template of NN (Dimension of input = 4)
        self.model.load_weights("double_pendulum.h5")
        self.weights = self.model.get_weights()
        self.mean, self.std = conf.init_scaler()
        
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
        q1_target = conf.q1_target       # Target position of the first link  
        q2_target = conf.q2_target       # Target position of the second link  


        # Cost definition
        self.cost_1 = 0
        self.cost_2 = 0
        self.running_costs_1 = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        self.running_costs_2 = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs_1[i] = self.w_v1 * v1[i]*v1[i]
            self.running_costs_1[i] += self.w_q1 * (q1[i] - q1_target)**2 
            self.running_costs_2[i] = self.w_v2 * v2[i]*v2[i]
            self.running_costs_2[i] += self.w_q2 * (q2[i] - q2_target)**2
            
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs_1[i] += self.w_u1 * u1[i]**2
                self.running_costs_1[i] += self.w_u2 * u2[i]**2

            self.cost_1 += self.running_costs_1[i]
            self.cost_2 += self.running_costs_2[i]

        self.opti.minimize(self.cost_1)
        self.opti.minimize(self.cost_2)

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
        q1n = (q1[self.N] - self.mean[0])/self.std[0]    # Normalize the state manually
        v1n = (v1[self.N] - self.mean[1])/self.std[1]    # Normalize the state manually
        q2n = (q2[self.N] - self.mean[2])/self.std[2]    # Normalize the state manually
        v2n = (v2[self.N] - self.mean[3])/self.std[3]    # Normalize the state manually
        state = np.array([q1n, v1n, q2n, v2n])
   
        if conf.terminal_constraint_on:
            self.opti.subject_to(self.nn_to_casadi(self.weights, state) > 1.2)

        # self.opti.callback(lambda i: print(i))

        return self.opti.solve()


if __name__ == "__main__":

    # Instance of MCP solver
    mpc = MpcDoublePendulum()
    
    initial_state = conf.initial_state     # Initial state of the pendulum (position and velocity)
    actual_trajectory = []                 # Lists to store the trajectory during the MPC iterations
    actual_inputs = []                     # Lists to store the inputs during the MPC iterations

    mpc_step = conf.mpc_step        # Number of MPC steps
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    # First run
    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[0]), sol.value(mpc.v1[0]), sol.value(mpc.q2[0]), sol.value(mpc.v2[0])]))
    actual_inputs.append((sol.value(mpc.u1[0]), sol.value(mpc.u2[0])))

    # We need to create a state_guess of size 4 x N+1
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q1[i+1])
        new_state_guess[1, i] = sol.value(mpc.v1[i+1])
        new_state_guess[2, i] = sol.value(mpc.q2[i+1])
        new_state_guess[3, i] = sol.value(mpc.v2[i+1])

    # We need to create an input_guess of size 2 x N
    for i in range(mpc.N-1):
        new_input_guess[0, i] = sol.value(mpc.u2[i+1])
        new_input_guess[1, i] = sol.value(mpc.u2[i+1])
    
    # Update the state and input guesses for the next MPC iteration
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
        
        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append((sol.value(mpc.u1[0]), sol.value(mpc.u2[0])))

        # Update state_guess for the next iteration
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q1[k+1])
            new_state_guess[1, k] = sol.value(mpc.v1[k+1])
            new_state_guess[2, k] = sol.value(mpc.q2[k+1])
            new_state_guess[3, k] = sol.value(mpc.v2[k+1])

        # Update input_guess for the next iteration
        for j in range(mpc.N-1):
            new_input_guess[0, j] = sol.value(mpc.u2[j+1])
            new_input_guess[1, j] = sol.value(mpc.u2[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
    
    # Initialize empty lists
    positions_q1 = []
    velocities_v1 = []
    positions_q2 = []
    velocities_v2 = []

    # Extract positions q1 and q2 from the actual trajectory 
    for i, state in enumerate(actual_trajectory):
        positions_q1.append(actual_trajectory[i][0])
        velocities_v1.append(actual_trajectory[i][1])
        positions_q2.append(actual_trajectory[i][2])
        velocities_v2.append(actual_trajectory[i][3])

    # Plot positions q1 and q2
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positions_q1, velocities_v1, c='blue', label='Pendulum 1 (q1)')
    ax.scatter(positions_q2, velocities_v2, c='red', label='Pendulum 2 (q2)')
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    ax.legend()
    plt.show()
    
    # Initialize empty lists
    positions_q1 = []
    velocities_v1 = []
    positions_q2 = []
    velocities_v2 = []
    torques_u1 = []
    torques_u2 = []

    # Extract data from the actual trajectory for a double pendulum
    for element in actual_trajectory:
        positions_q1.append(element[0])
        velocities_v1.append(element[1])
        positions_q2.append(element[2])
        velocities_v2.append(element[3])

    # Extract torques for each pendulum
    for element in actual_inputs:
        torques_u1.append(element[0])
        torques_u2.append(element[1])
    
    # Plot position for both pendulums
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(positions_q1, label='Pendulum 1')
    ax.plot(positions_q2, label='Pendulum 2')
    ax.set_xlabel('mpc step')
    ax.set_ylabel('q [rad]')
    ax.set_title('Position')
    ax.legend()
    plt.show()

    # Plot velocity for both pendulums
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(velocities_v1, label='Pendulum 1')
    ax.plot(velocities_v2, label='Pendulum 2')
    ax.set_xlabel('mpc step')
    ax.set_ylabel('v [rad/s]')
    ax.set_title('Velocity')
    ax.legend()
    plt.show()

    # Plot torques for both pendulums
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(torques_u1, label='Torque 1')
    ax.plot(torques_u2, label='Torque 2')
    ax.set_xlabel('mpc step')
    ax.set_ylabel('u [N/m]')
    ax.set_title('Torque')
    ax.legend()
    plt.show()

    # Create DataFrames for positions and torques
    columns_positions = ['Positions_q1', 'Positions_q2']
    df_positions = pd.DataFrame({'Positions_q1': positions_q1, 'Positions_q2': positions_q2}, columns=columns_positions)
    
    # Export DataFrames to csv format
    df_positions.to_csv('../DoublePendulum/Plots_&_Animations/Position_double.csv', index=False)