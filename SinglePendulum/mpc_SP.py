import numpy as np
import casadi
import SP_dynamics as dynamics
import mpc_SP_conf as config
from nn_SP_relu import create_model_relu
from nn_SP_tanh import create_model_tanh
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class MpcSinglePendulum:

    def __init__(self):
        self.N = config.N                     # MPC horizon
        self.w_q = config.w_q                 # Position weight
        self.w_u = config.w_u                 # Input weight
        self.w_v = config.w_v                 # Velocity weight
        
        #self.model = create_model_relu(2)        # Template of NN with ReLu
        self.model = create_model_tanh(2)       # Template of NN with Tanh
        self.model.load_weights(config.nn)
        self.weights = self.model.get_weights()
        self.mean_x, self.std_x, self.mean_y, self.std_y = config.init_scaler()
    
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
        out = np.array(x)  # Initialize the output as the normalized input (state vector)
        it = 0  # Counter to distinguish between weights and biases

        for param in params:
            param = np.array(param.tolist())  # Convert the parameter to a numpy array

            if it % 2 == 0:
                # If it's a weight layer, perform the product between the current output and the weights
                out = out @ param
            else:
                # If it's a bias layer, add the biases
                out = param + out
                
                # Apply the 'tanh' activation function for every layer except the last one
                if it < len(params) - 2:  # Skip the last layer (linear output)
                    for i, item in enumerate(out):
                        out[i] = casadi.tanh(casadi.MX(out[i]))

            it += 1

        return casadi.MX(out[0])  # Return the final output as a CasADi symbol

    
    def solve(self, x_init, X_guess = None, U_guess = None):
        self.opti = casadi.Opti()  # Initialize the optimizer
        
        # Declaration of the variables in casaDi types
        self.q = self.opti.variable(self.N+1)    # States
        self.v = self.opti.variable(self.N+1)    # Velocities
        self.u = self.opti.variable(self.N)      # Control inputs

        # Alias variables for convenience
        q = self.q
        v = self.v
        u = self.u
        
        # Target position
        q_target = config.q_target

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
        s_opst = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)
 
        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            self.running_costs[i] += self.w_q * (q[i] - q_target)**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        
        # Adding terminal cost from the neural network
        if config.TC:
            state = [(q[self.N] - self.mean_x[0]) / self.std_x[0], (v[self.N] - self.mean_x[1]) / self.std_x[1]]
            self.terminal_cost = self.nn_to_casadi(self.weights, state)
            self.cost += self.terminal_cost

        # Minimize the total cost
        self.opti.minimize(self.cost)

        # Dynamics Constraint
        for i in range(self.N):
            x_next = dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])

        # Initial state constraint       
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # State bound constraints
        if config.costraint:
            for i in range(self.N+1):
                self.opti.subject_to(q[i] <= config.q_min)
                self.opti.subject_to(q[i] >= config.q_max)
                self.opti.subject_to(v[i] <= config.v_min)
                self.opti.subject_to(v[i] >= config.v_max)
                #self.opti.subject_to(self.opti.bounded(config.q_min, q[i], config.q_max))
                #self.opti.subject_to(self.opti.bounded(config.v_min, v[i], config.v_max))
            for i in range(self.N):
                self.opti.subject_to(u[i] <= config.u_min)
                self.opti.subject_to(u[i] >= config.u_max)
                #self.opti.subject_to(self.opti.bounded(config.u_min, u[i], config.u_max))


        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)

        # Solve the optimization problem
        solv = self.opti.solve()

        # === Debug window ===
        current_state = np.array([solv.value(q[0]), solv.value(v[0])])  # Current state
        rescaled_state = np.array([(current_state[0] - self.mean_x[0]) / self.std_x[0], (current_state[1] - self.mean_x[1]) / self.std_x[1]])  # Rescaled state

        print("==== MPC Step Debug Info ====")
        print(f"Current state: q = {current_state[0]:.4f}, v = {current_state[1]:.4f}")
        print(f"Rescaled state: q' = {rescaled_state[0]:.4f}, v' = {rescaled_state[1]:.4f}")

        # Display running costs and states
        print(f"Running costs for each step:")
        for i in range(self.N+1):
            state_q = solv.value(q[i])
            state_v = solv.value(v[i])
            running_cost = solv.value(self.running_costs[i])
            print(f"Step {i}: q = {state_q:.4f}, v = {state_v:.4f}, Running cost = {running_cost:.4f}")

        # Terminal cost (only if config.TC == 1)
        if config.TC:
            terminal_q = solv.value(q[self.N])
            terminal_v = solv.value(v[self.N])
            terminal_cost_value = solv.value(self.terminal_cost)
            print(f"Terminal state: q = {terminal_q:.4f}, v = {terminal_v:.4f}")
            print(f"Terminal cost from NN: {terminal_cost_value:.4f}, calculated at state {[solv.value(q[self.N]), solv.value(v[self.N])]}")

        # Total cost
        print(f"Total cost: {solv.value(self.cost):.4f}")
        print("=============================")

        return solv

def plot_and_save(data, xlabel, ylabel, title, filename):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.grid(True)
    plt.close(fig)

if __name__ == "__main__":

    # Instance of OCP solver
    mpc = MpcSinglePendulum()
    
    constr_status = "constrained" if config.costraint else "unconstrained"
    print("==========================")
    print(f"You're running the following test: target: {config.q_target}, "
          f"initial state: {config.initial_state}, {constr_status} problem, TC = {config.TC}, N = {config.N}, T = {config.T}")
    
    initial_state = config.initial_state     # Initial state of the pendulum (position and velocity)
    actual_trajectory = []      # Buffer to store actual trajectory
    actual_inputs = []          # Buffer to store actual inputs
    total_costs = []            # Buffer to store total costs
    terminal_costs = []         # Buffer to store terminal costs

    mpc_step = config.mpc_step          # Number of MPC steps
    new_state_guess = np.zeros((2, mpc.N+1))
    new_input_guess = np.zeros((mpc.N))

    # Start timer
    start_time = time.time()
    
    # First run
    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
    actual_inputs.append(sol.value(mpc.u[0]))

    # Creation of state_guess of size 2 x N+1
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q[i+1])
        new_state_guess[1, i] = sol.value(mpc.v[i+1])

    # Creation of input_guess of size N
    for i in range(mpc.N-1):
        new_input_guess[i] = sol.value(mpc.u[i+1])
    
    # Update the state and input guesses for the next MPC iteration
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
        if config.TC:
            terminal_costs.append(sol.value(mpc.terminal_cost))

        # Update state_guess for the next iteration
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q[k+1])
            new_state_guess[1, k] = sol.value(mpc.v[k+1])

        # Update input_guess for the next iteration
        for j in range(mpc.N-1):
            new_input_guess[j] = sol.value(mpc.u[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        
    # Stop timer
    end_time = time.time()
    
    # Time in nice format
    tot_time = end_time - start_time
    hours = int(tot_time / 3600)
    minutes = int((tot_time - 3600*hours) / 60)       
    seconds = tot_time - hours*3600 - minutes*60
    print("Total elapsed time: {}h {}m {:.2f}s".format(hours, minutes, seconds))

    # Initialize empty lists
    positions = []
    velocities = []

    # Extract positions and velocities from the actual trajectory
    for i, state in enumerate(actual_trajectory):
        positions.append(actual_trajectory[i][0])
        velocities.append(actual_trajectory[i][1])

    # Create directory for plots
    output_dir = 'Plots_&_Animations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the suffix from the file name
    suffix = config.nn.split('_SP_')[-1].replace('.h5', '')

    if config.TC:
        test_dir = os.path.join(output_dir, f'Test_{suffix}_TC')
    else:
        test_dir = os.path.join(output_dir, f'Test_{suffix}_NTC')

    # Create dedicate directory for the test
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Plot and save results
    plot_and_save(positions, 'mpc step', 'q [rad]', 'Position', os.path.join(test_dir, f'position_plot_{suffix}.png'))
    plot_and_save(velocities, 'mpc step', 'v [rad/s]', 'Velocity', os.path.join(test_dir, f'velocity_plot_{suffix}.png'))
    plot_and_save(actual_inputs, 'mpc step', 'u [N/m]', 'Torque', os.path.join(test_dir, f'torque_plot_{suffix}.png'))
    plot_and_save(total_costs, 'MPC Step', 'Total Cost', 'Total Cost', os.path.join(test_dir, f'total_cost_plot_{suffix}.png'))
    plot_and_save(terminal_costs, 'MPC Step', 'Terminal Cost', 'Terminal Cost', os.path.join(test_dir, f'terminal_cost_plot_{suffix}.png'))

    # Save data in a .csv file
    if (config.TC):
        mpc.save_results(positions, velocities, actual_inputs, total_costs, terminal_costs, f'Plots_&_Animations/mpc_SP_TC_{suffix}.csv')
    else:
        mpc.save_results(positions, velocities, actual_inputs, total_costs, terminal_costs, f'Plots_&_Animations/mpc_SP_NTC_{suffix}.csv')