import numpy as np
import casadi
import SP_dynamics as dynamics
import mpc_SP_conf as config
from nn_SP import create_model
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
        
        self.model = create_model(2)        # Template of NN
        self.model.load_weights(config.nn)
        self.weights = self.model.get_weights()
        self.mean_x, self.std_x, self.mean_y, self.std_y = config.init_scaler()

        # Print mean and std deviation for scaling
        print("Scaler X Mean:", self.mean_x)
        print("Scaler X std:", self.std_x)
        print("Scaler y Mean:", self.mean_y)
        print("Scaler y std:", self.std_y)
    
    def save_results(self, positions, velocities, actual_inputs, total_costs, filename):        
        total_costs.extend([None] * (len(positions) - len(total_costs)))
        df = pd.DataFrame({
            'Positions': positions,
            'Velocities': velocities,
            'Inputs': actual_inputs,
            'Total_Costs': total_costs,
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
    

    def solve(self, x_init, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()  # Initialize the optimizer

        # Declaration of the variables in CasADi types
        self.q = self.opti.variable(self.N+1)    # States (position)
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
                self.opti.set_initial(q[i], X_guess[0, i])
                self.opti.set_initial(v[i], X_guess[1, i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q[i], x_init[0])
                self.opti.set_initial(v[i], x_init[1])
        
        # Control input vector initialization
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u[i], U_guess[i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)
        
        # Cost definition
        self.cost = 0
        self.running_costs = [None,] * (self.N)  # Vector to hold running cost values for each step
        for i in range(self.N):
            self.running_costs[i] = self.w_v * v[i]**2
            self.running_costs[i] += self.w_q * (q[i] - q_target)**2
            if i < self.N:  # No input cost at the last step
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]

        # Minimize the total cost
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            x_next = dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])

        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])
        
        if config.costraint:
            # State bound constraints
            for i in range(self.N+1):
                self.opti.subject_to(q[i] >= config.q_min)
                self.opti.subject_to(q[i] <= config.q_max)
                self.opti.subject_to(v[i] >= config.v_min)
                self.opti.subject_to(v[i] <= config.v_max)

            # Input bound constraints
            for i in range(self.N):
                self.opti.subject_to(u[i] >= config.u_min)
                self.opti.subject_to(u[i] <= config.u_max)
        
        # Solve the optimization problem
        solv = self.opti.solve()

        # === Debug window ===
        current_state = np.array([solv.value(q[0]), solv.value(v[0])])  # Current state

        print("==== MPC Step Debug Info ====")
        print(f"Current state: q = {current_state[0]:.4f}, v = {current_state[1]:.4f}")

        # Display running costs and states
        print(f"Running costs for each step:")
        for i in range(self.N):
            state_q = solv.value(q[i])
            state_v = solv.value(v[i])
            running_cost = solv.value(self.running_costs[i])
            print(f"Step {i}: q = {state_q:.4f}, v = {state_v:.4f}, Running cost = {running_cost:.4f}")

        # Total cost
        print(f"Total cost: {solv.value(self.cost):.4f}")
        print("=============================")

        return solv

if __name__ == "__main__":
    # Load initial states from CSV files
    csv_files = ['Plots_&_Animations/Test_225_unconstr_tanh_test_1/mpc_SP_TC_225_unconstr_tanh_test_1.csv', 
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_2/mpc_SP_TC_225_unconstr_tanh_test_2.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_3/mpc_SP_TC_225_unconstr_tanh_test_3.csv', 
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_4/mpc_SP_TC_225_unconstr_tanh_test_4.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_5/mpc_SP_TC_225_unconstr_tanh_test_5.csv', 
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_6/mpc_SP_TC_225_unconstr_tanh_test_6.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_7/mpc_SP_TC_225_unconstr_tanh_test_7.csv', 
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_9/mpc_SP_TC_225_unconstr_tanh_test_9.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_10/mpc_SP_TC_225_unconstr_tanh_test_10.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_11/mpc_SP_TC_225_unconstr_tanh_test_11.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_12/mpc_SP_TC_225_unconstr_tanh_test_12.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_13/mpc_SP_TC_225_unconstr_tanh_test_13.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_14/mpc_SP_TC_225_unconstr_tanh_test_14.csv',
                 'Plots_&_Animations/Test_225_unconstr_tanh_test_15/mpc_SP_TC_225_unconstr_tanh_test_15.csv'] 
     
    initial_states = []

    for file in csv_files:
        # Read only the first row of each CSV
        df = pd.read_csv(file, nrows=1)
        initial_states.append(df[['Positions', 'Velocities']].values[0])

    # Convert the list of initial states into a NumPy array
    initial_states = np.array(initial_states)

    num_tests = len(initial_states)  # Number of tests based on the number of CSV files

    # Instance of OCP solver
    mpc = MpcSinglePendulum()

    constr_status = "constrained" if config.costraint else "unconstrained"
    print("==========================")
    print(f"You're running the following test: target: {config.q_target}, {constr_status} problem, TC = {config.TC}, N = {config.N}")

    # Create main directory for plots
    output_dir = 'Plots_&_Animations_NTC'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start timer
    start_time = time.time()

    for test_idx, initial_state in enumerate(initial_states):
        print(f"\n========== Test {test_idx+1}/{num_tests} ==========")
        print(f"Initial state: q = {initial_state[0]:.4f}, v = {initial_state[1]:.4f}")

        # Create directory for each test
        suffix = config.nn.split('_SP_')[-1].replace('.h5', '')
        test_dir = os.path.join(output_dir, f'Test_{suffix}_test_{test_idx+1}')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        actual_trajectory = []      # Buffer to store actual trajectory
        actual_inputs = []          # Buffer to store actual inputs
        total_costs = []            # Buffer to store total costs

        mpc_step = config.mpc_step          # Number of MPC steps
        new_state_guess = np.zeros((2, mpc.N+1))
        new_input_guess = np.zeros((mpc.N))

        # First run
        sol = mpc.solve(initial_state)
        actual_trajectory.append(np.array([sol.value(mpc.q[0]), sol.value(mpc.v[0])]))
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

        # Plot and save results for the current test
        test_suffix = f'{suffix}_test_{test_idx+1}'

        # Save data in a .csv file for the current test
        if ((config.TC == 1) and (config.noise == 1)):
            mpc.save_results(positions, velocities, actual_inputs, total_costs, f'{test_dir}/mpc_SP_TC_noise_{test_suffix}.csv')
        if ((config.TC == 1) and (config.noise == 0)):
            mpc.save_results(positions, velocities, actual_inputs, total_costs, f'{test_dir}/mpc_SP_TC_{test_suffix}.csv')
        if((config.TC == 0) and (config.scenario_type == 1)):
            mpc.save_results(positions, velocities, actual_inputs, total_costs, f'{test_dir}/mpc_SP_NTC_T_1_{test_suffix}.csv')
        if((config.TC == 0) and (config.scenario_type == 0)):
            mpc.save_results(positions, velocities, actual_inputs, total_costs, f'{test_dir}/mpc_SP_NTC_T_0.01_{test_suffix}.csv')
