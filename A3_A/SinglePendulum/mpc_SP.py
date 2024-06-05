import numpy as np
import casadi
import SP_dynamics as SP_dynamics
import mpc_SP_conf as conf
from nn_SP_TensorFlow import create_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class MpcSinglePendulum:

    def __init__(self):
        self.N = conf.N                 # MPC horizon
        self.w_q = conf.w_q             # Position weight
        self.w_u = conf.w_u             # Input weight
        self.w_v = conf.w_v             # Velocity weight
        
        self.model = create_model(2)        # Template of NN
        self.model.load_weights("nn_SP_2.h5")
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
    
    def solve(self, x_init, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()  # Initialize the optimizer
        
        # Create vectors for states and control inputs
        # Size of state vector is N+1, as it includes initial and final states
        # Size of control input is N, since the final control input is not significant
        self.q = self.opti.variable(self.N+1)    # States
        self.v = self.opti.variable(self.N+1)    # Velocities
        self.u = self.opti.variable(self.N)      # Control inputs

        # Alias variables for convenience
        q, v, u = self.q, self.v, self.u
        
        # Target position
        q_target = conf.q_target

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

        # State Normalization 
        # state = [(q[self.N] - self.mean[0])/self.std[0], (v[self.N] - self.mean[1])/self.std[1]]
        state = [q[self.N], v[self.N]]

        # Cost definition
        self.cost = 0
        self.running_costs = [None] * (self.N+1)
        self.terminal_cost = self.nn_to_casadi(self.weights, state)
        #self.terminal_cost1 = self.nn_to_casadi(self.weights, state1)
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            self.running_costs[i] += self.w_q * (q[i] - q_target)**2
            if (i<self.N):   # Check necessary since at the last step it doesn't make sense to consider the input                        
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        
        # Adding terminal cost from the neural network
        if (conf.TC):
            self.cost += self.terminal_cost    
        self.opti.minimize(self.cost)

        # Dynamics Constraint
        for i in range(self.N):
            x_next = SP_dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])

        # Initial state constraint       
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # State bound constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.opti.bounded(conf.q_min, q[i], conf.q_max))
            self.opti.subject_to(self.opti.bounded(conf.v_min, v[i], conf.v_max))
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(conf.u_min, u[i], conf.u_max))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)
        sol = self.opti.solve()

        # Stampa dei costi in esecuzione e del costo terminale
        #for i in range(self.N+1):
        #    running_cost_value = float(sol.value(self.running_costs[i]))
        #    print(f"Running Cost at step {i}: {running_cost_value:.6f}")
        #terminal_cost_value = float(sol.value(self.terminal_cost))
        #print(f"Terminal Cost: {terminal_cost_value:.6f}")
        #total_cost_value = float(sol.value(self.cost))
        #print(f"Total Cost: {total_cost_value:.6f}")

        return sol


if __name__ == "__main__":
    # Instance of MCP solver
    mpc = MpcSinglePendulum()

    initial_state = conf.initial_state
    actual_trajectory = []
    actual_inputs = []
    total_costs = []
    terminal_costs = []

    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((2, mpc.N+1))
    new_input_guess = np.zeros((mpc.N))

    # Start timer
    start_time = time.time()

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

        actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
        actual_inputs.append(sol.value(mpc.u[0]))
        total_costs.append(sol.value(mpc.cost))
        terminal_costs.append(sol.value(mpc.terminal_cost))
        
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q[k+1])
            new_state_guess[1, k] = sol.value(mpc.v[k+1])
        for j in range(mpc.N-1):
            new_input_guess[j] = sol.value(mpc.u[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_costs[0]), "Terminal Cost", sol.value(mpc.terminal_cost))
    
    # Stop timer
    end_time = time.time()
    
    # Time in nice format
    tot_time = end_time - start_time
    hours = int(tot_time / 3600)
    minutes = int((tot_time - 3600*hours) / 60)       
    seconds = tot_time - hours*3600 - minutes*60
    print("Total elapsed time: {}h {}m {:.2f}s".format(hours, minutes, seconds))

    positions = []
    velocities = []

    for element in actual_trajectory:
        positions.append(element[0])
        velocities.append(element[1])

    output_dir = 'Plots_&_Animations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Position plot
    fig = plt.figure(figsize=(12, 8))
    plt.plot(positions)
    plt.xlabel('mpc step')
    plt.ylabel('q [rad]')
    plt.title('Position')
    plt.savefig(os.path.join(output_dir, 'position_plot.png'))
    plt.close(fig)

    # Velocity plot
    fig = plt.figure(figsize=(12, 8))
    plt.plot(velocities)
    plt.xlabel('mpc step')
    plt.ylabel('v [rad/s]')
    plt.title('Velocity')
    plt.savefig(os.path.join(output_dir, 'velocity_plot.png'))
    plt.close(fig)

    # Torque plot
    fig = plt.figure(figsize=(12, 8))
    plt.plot(actual_inputs)
    plt.xlabel('mpc step')
    plt.ylabel('u [N/m]')
    plt.title('Torque')
    plt.savefig(os.path.join(output_dir, 'torque_plot.png'))
    plt.close(fig)

    # Total cost plot
    fig = plt.figure(figsize=(12, 8))
    plt.plot(total_costs)
    plt.xlabel('MPC Step')
    plt.ylabel('Total Cost')
    plt.title('Total Cost')
    plt.savefig(os.path.join(output_dir, 'total_cost_plot.png'))
    plt.close(fig)

    # Terminal cost plot
    fig = plt.figure(figsize=(12, 8))
    plt.plot(terminal_costs)
    plt.xlabel('MPC Step')
    plt.ylabel('Terminal Cost')
    plt.title('Terminal Cost')
    plt.savefig(os.path.join(output_dir, 'terminal_cost_plot.png'))
    plt.close(fig)

    # DataFrame starting from the final array
    columns_positions = ['Pos_q1']
    df = pd.DataFrame(positions, columns=columns_positions)

    # Export DataFrame to csv format
    df.to_csv('/home/student/shared/orc/A3_A/SinglePendulum/Plots_&_Animations/SinglePendulum.csv', index=False)