import numpy as np
import casadi
import SP_dynamics as dynamics
import mpc_SP_conf as config
from nn_SP_TensorFlow import create_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class MpcSinglePendulum:

    def __init__(self):
        self.N = config.N                     # MPC horizon
        self.weight_position = config.w_q     # Position weight
        self.weight_input = config.w_u        # Input weight
        self.weight_velocity = config.w_v     # Velocity weight

        self.model = create_model(2)          # Template of NN
        self.model.load_weights(config.nn)
        self.weights = self.model.get_weights()
        self.mean_x, self.std_x, self.mean_y, self.std_y = config.init_scaler()

        # Print to verify the initialization
        print("mean_x:", self.mean_x)
        print("std_x:", self.std_x)
        print("mean_y:", self.mean_y)
        print("std_y:", self.std_y)
    
    def save_results(self, positions, velocities, actual_inputs, total_costs, terminal_costs, filename):        
        total_costs.extend([None] * (len(positions) - len(total_costs)))
        terminal_costs.extend([None] * (len(positions) - len(terminal_costs)))
        df = pd.DataFrame({
            'Positions': positions,
            'Velocities': velocities,
            'Inputs': actual_inputs,
            'Total_Costs': total_costs,
            'Terminal_Costs': terminal_costs
        })
        df.to_csv(filename, index=False)
    
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
                    #out[i] = casadi.MX(out[i])

            it += 1

        return casadi.MX(out[0])

    def solve(self, x_init, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()  # Initialize the optimizer
        
        # Declaration of the variables in casaDi types
        self.q = self.opti.variable(self.N+1)    # States
        self.v = self.opti.variable(self.N+1)    # Velocities
        self.u = self.opti.variable(self.N)      # Control inputs

        # Alias variables for convenience
        q, v, u = self.q, self.v, self.u
        
        # Target position
        q_target = config.q_target

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
        state = [(q[self.N] - self.mean_x[0])/self.std_x[0], (v[self.N] - self.mean_x[1])/self.std_x[1]]
        self.terminal_cost = self.nn_to_casadi(self.weights, state)

        # Cost definition
        self.total_cost = 0
        self.running_costs = [None] * (self.N+1)
        for i in range(self.N+1):
            self.running_costs[i] = self.weight_velocity * v[i]**2
            self.running_costs[i] += self.weight_position * (q[i] - q_target)**2
            if (i < self.N):                        
                self.running_costs[i] += self.weight_input * u[i]**2
            self.total_cost += self.running_costs[i]
        
        # Adding terminal cost from the neural network
        if (config.TC):
            self.total_cost += self.terminal_cost
        self.opti.minimize(self.total_cost)

        # Dynamics Constraint
        for i in range(self.N):
            x_next = dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])

        # Initial state constraint       
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # State bound constraints
        for i in range(self.N+1):
            self.opti.subject_to(self.opti.bounded(config.q_min, q[i], config.q_max))
            self.opti.subject_to(self.opti.bounded(config.v_min, v[i], config.v_max))
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(config.u_min, u[i], config.u_max))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)
        
        # Solve the optimization problem
        solv = self.opti.solve()

        # Stampa dei costi in esecuzione e del costo terminale
        for i in range(self.N+1):
            running_cost_value = float(solv.value(self.running_costs[i]))
            print(f"Running Cost at step {i}: {running_cost_value:.6f}")
        state_print = [solv.value(self.q)[self.N], solv.value(self.v)[self.N]]
        print("State:", state_print)
        state_print_normalized = [(solv.value(self.q)[self.N-1]-self.mean_x[0])/self.std_x[0], (solv.value(self.v)[self.N-1]-self.mean_x[1])/self.std_x[1]]
        print("Normalized state:", state_print_normalized)
        terminal_cost_value = float(solv.value(self.terminal_cost))
        print(f"Terminal Cost: {terminal_cost_value:.6f}")
        total_cost_value = float(solv.value(self.total_cost))
        print(f"Total Cost: {total_cost_value:.6f}")

        return self.opti.solve()
    
def plot_and_save(data, xlabel, ylabel, title, filename):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close(fig)

if __name__ == "__main__":
    # Instance of MCP solver
    mpc_solver = MpcSinglePendulum()

    initial_state = config.initial_state
    actual_trajectory = []      # Buffer to store actual trajectory
    actual_inputs = []          # Buffer to store actual inputs
    total_costs = []            # Buffer to store total costs
    terminal_costs = []         # Buffer to store terminal costs

    mpc_step = config.mpc_step
    new_state_guess = np.zeros((2, mpc_solver.N+1))
    new_input_guess = np.zeros((mpc_solver.N))

    # Start timer
    start_time = time.time()

    sol = mpc_solver.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc_solver.q[1]), sol.value(mpc_solver.v[1])]))
    actual_inputs.append(sol.value(mpc_solver.u[0]))
    
    for i in range(mpc_solver.N):
        new_state_guess[0, i] = sol.value(mpc_solver.q[i+1])
        new_state_guess[1, i] = sol.value(mpc_solver.v[i+1])
    for i in range(mpc_solver.N-1):
        new_input_guess[i] = sol.value(mpc_solver.u[i+1])
        
    for i in range(mpc_step):
        init_state = actual_trajectory[i]

        try:
            sol = mpc_solver.solve(init_state, new_state_guess, new_input_guess)
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("\n ###########################################")
                print("#  MPC stopped due to infeasible problem  #")
                print("########################################### \n")
                print(mpc_solver.opti.debug.show_infeasibilities())
                break
            else:
                print(e)

        actual_trajectory.append(np.array([sol.value(mpc_solver.q[1]), sol.value(mpc_solver.v[1])]))
        actual_inputs.append(sol.value(mpc_solver.u[0]))
        total_costs.append(sol.value(mpc_solver.total_cost))
        terminal_costs.append(sol.value(mpc_solver.terminal_cost))
      
        for k in range(mpc_solver.N):
            new_state_guess[0, k] = sol.value(mpc_solver.q[k+1])
            new_state_guess[1, k] = sol.value(mpc_solver.v[k+1])
        for j in range(mpc_solver.N-1):
            new_input_guess[j] = sol.value(mpc_solver.u[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        if (config.TC):
            print("Cost", sol.value(mpc_solver.total_cost), "Running cost", sol.value(mpc_solver.running_costs[0]), "Terminal Cost", sol.value(mpc_solver.terminal_cost))
        else: 
            print("Cost", sol.value(mpc_solver.total_cost), "Running cost", sol.value(mpc_solver.running_costs[0]))
    
    # Stop timer
    end_time = time.time()
    
    # Time in nice format
    tot_time = end_time - start_time
    hours = int(tot_time / 3600)
    minutes = int((tot_time - 3600*hours) / 60)       
    seconds = tot_time - hours*3600 - minutes*60
    print("Total elapsed time: {}h {}m {:.2f}s".format(hours, minutes, seconds))

    # Extract positions and velocities from the actual trajectory
    positions = [state[0] for state in actual_trajectory]
    velocities = [state[1] for state in actual_trajectory]

    # Generate grid of states
    _, state_array = config.grid_states(121, 121)
    to_test = config.scaler_X.fit_transform(state_array)

    # Predict costs using the neural network
    cost_pred = mpc_solver.model.predict(to_test)
    
    # De-standardize the predictions to obtain the results in the original scale
    predicted_costs = config.scaler_y.inverse_transform(cost_pred)
    #predicted_costs = cost_pred * mpc_solver.std_y[0] + mpc_solver.mean_y[0]

    # Plotting the results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()

    # Create a scatter plot with colormap based on predicted costs
    sc = ax.scatter(state_array[:, 0], state_array[:, 1], c=predicted_costs, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Predicted Cost')

    # Overlay the trajectory on the colormap
    ax.scatter(positions, velocities, color='red', s=30, label='Trajectory')

    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    ax.legend()
    plt.title('State Space with Predicted Costs and Trajectory')
    plt.show()

    output_dir = 'Plots_&_Animations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot and save results
    plot_and_save(positions, 'mpc step', 'q [rad]', 'Position', os.path.join(output_dir, 'position_plot.png'))
    plot_and_save(velocities, 'mpc step', 'v [rad/s]', 'Velocity', os.path.join(output_dir, 'velocity_plot.png'))
    plot_and_save(actual_inputs, 'mpc step', 'u [N/m]', 'Torque', os.path.join(output_dir, 'torque_plot.png'))
    plot_and_save(total_costs, 'MPC Step', 'Total Cost', 'Total Cost', os.path.join(output_dir, 'total_cost_plot.png'))
    plot_and_save(terminal_costs, 'MPC Step', 'Terminal Cost', 'Terminal Cost', os.path.join(output_dir, 'terminal_cost_plot.png'))

    # Save data in a .csv file
    mpc_solver.save_results(positions, velocities, actual_inputs, total_costs, terminal_costs, 'Plots_&_Animations/MPC_SinglePendulum_NTC.csv')