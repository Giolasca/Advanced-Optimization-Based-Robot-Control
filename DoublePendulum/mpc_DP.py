import numpy as np
import casadi
import DP_dynamics as DP_dynamics
import mpc_DP_conf as conf
from nn_DP import create_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class MpcDoublePendulum:

    def __init__(self):
        self.N = conf.N                     # Number of steps
        self.w_q1 = conf.w_q1                 # Position weight first link 
        self.w_u1 = conf.w_u1                 # Input weight first link
        self.w_v1 = conf.w_v1                 # Velocity weight first link

        self.w_q2 = conf.w_q2                 # Position weight second link
        self.w_u2 = conf.w_u2                 # Input weight second link
        self.w_v2 = conf.w_v2                 # Velocity weight second link

        self.model = create_model(4)        # Template of NN for double pendulum
        self.model.load_weights(conf.nn)
        self.weights = self.model.get_weights()
        self.mean_x, self.std_x, self.mean_y, self.std_y = conf.init_scaler()
    

    def save_results(self, pos1, vel1, pos2, vel2, actual_inputs, total_costs, terminal_costs, true_tc, terminal_cost_error, filename):
        max_length = max(len(pos1), len(vel1), len(pos2), len(vel2), len(actual_inputs), len(total_costs), len(terminal_costs))
        pos1.extend([None] * (max_length - len(pos1)))
        vel1.extend([None] * (max_length - len(vel1)))
        pos2.extend([None] * (max_length - len(pos2)))
        vel2.extend([None] * (max_length - len(vel2)))
        actual_inputs.extend([None] * (max_length - len(actual_inputs)))
        total_costs.extend([None] * (max_length - len(total_costs)))
        
        # Create the base dictionary for the DataFrame
        data = {
            'q1': pos1,
            'v1': vel1,
            'q2': pos2,
            'v2': vel2,
            'u1': [u[0] if u is not None else None for u in actual_inputs],
            'u2': [u[1] if u is not None else None for u in actual_inputs],
            'Total_Costs': total_costs,
        }

        # Add optional columns if `config.TC` is enabled
        if conf.TC:      
            terminal_costs.extend([None] * (max_length - len(terminal_costs)))
            true_tc.extend([None] * (len(pos1) - len(true_tc)))
            terminal_cost_error.extend([None] * (len(pos1) - len(terminal_cost_error)))
            data.update({
                'Terminal_Costs': terminal_costs,
                'True_Terminal_Costs': true_tc,
                'Terminal_Cost_Error': terminal_cost_error
            })

        # Create the DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def calculate_true_terminal_cost(self, final_state):
        q1_N = final_state[0]
        v1_N = final_state[1]
        q2_N = final_state[2]
        v2_N = final_state[3]

        # Define the terminal cost 
        pos_cost = self.w_q1 * (q1_N - conf.q1_target)**2 + self.w_q2 * (q2_N - conf.q2_target)**2
        vel_cost = self.w_v1 * v1_N**2 + self.w_v2 * v2_N**2
        true_terminal_cost = pos_cost + vel_cost

        return true_terminal_cost


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

        return casadi.fabs(casadi.MX(out[0]))  # Return the final output as a CasADi symbol
    
    
    def solve(self, x_init, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()  # Initialize the optimizer
        
        # Declaration of the variables in casaDi types
        self.q1 = self.opti.variable(self.N+1)
        self.v1 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        q1, v1, u1 = self.q1, self.v1, self.u1
        
        self.q2 = self.opti.variable(self.N+1)
        self.v2 = self.opti.variable(self.N+1)
        self.u2 = self.opti.variable(self.N)
        q2, v2, u2 = self.q2, self.v2, self.u2

        # Target positions
        q1_target, q2_target = conf.q1_target, conf.q2_target
        
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

        # Cost definition
        self.cost = 0
        self.running_costs = [None] * (self.N+1)
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            self.running_costs[i] += self.w_q1 * (q1[i] - q1_target)**2 + self.w_q2 * (q2[i] - q2_target)**2
            if (i<self.N):   
                self.running_costs[i] += self.w_u1 * u1[i]**2 + self.w_u1 * u2[i]**2
            self.cost += self.running_costs[i] 

        # Adding terminal cost from the neural network
        if (conf.TC == 1):
            state = [(q1[self.N] - self.mean_x[0])/self.std_x[0], (v1[self.N] - self.mean_x[1])/self.std_x[1], 
                 (q2[self.N] - self.mean_x[2])/self.std_x[2], (v2[self.N] - self.mean_x[3])/self.std_x[3]]
            self.terminal_cost = self.nn_to_casadi(self.weights, state)
            self.cost += self.terminal_cost    

        # Minimize the total cost
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            x_next = DP_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(v1[i+1] == x_next[1])
            self.opti.subject_to(q2[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])

        # Initial state constraint       
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])

        if conf.costraint:
            # Bounds constraints
            for i in range(self.N+1):
                self.opti.subject_to(self.opti.bounded(conf.q1_min, q1[i], conf.q1_max))
                self.opti.subject_to(self.opti.bounded(conf.v1_min, v1[i], conf.v1_max))
                self.opti.subject_to(self.opti.bounded(conf.q2_min, q2[i], conf.q2_max))
                self.opti.subject_to(self.opti.bounded(conf.v1_min, v2[i], conf.v2_max))
            for i in range(self.N):
                self.opti.subject_to(self.opti.bounded(conf.u1_min, u1[i], conf.u1_max))
                self.opti.subject_to(self.opti.bounded(conf.u2_min, u2[i], conf.u2_max))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)

        # Solve the optimization problem
        solv = self.opti.solve()

        # === Debug window ===
        current_state = np.array([solv.value(q1[0]), solv.value(v1[0]), solv.value(q2[0]), solv.value(v2[0])])  # Current state
        rescaled_state = np.array([(current_state[0] - self.mean_x[0]) / self.std_x[0], (current_state[1] - self.mean_x[1]) / self.std_x[1], (current_state[2] - self.mean_x[2]) / self.std_x[2], (current_state[3] - self.mean_x[3]) / self.std_x[3]])  # Rescaled state

        print("==== MPC Step Debug Info ====")
        print(f"Current state: q1 = {current_state[0]:.4f}, v1 = {current_state[1]:.4f}, q2 = {current_state[2]:.4f}, v2 = {current_state[3]:.4f}")
        print(f"Rescaled state: q1' = {rescaled_state[0]:.4f}, v1' = {rescaled_state[1]:.4f}, q2' = {rescaled_state[2]:.4f}, v2' = {rescaled_state[3]:.4f}")

        # Display running costs and states
        print(f"Running costs for each step:")
        for i in range(self.N+1):
            state_q1 = solv.value(q1[i])
            state_v1 = solv.value(v1[i])
            state_q2 = solv.value(q2[i])
            state_v2 = solv.value(v2[i])
            running_cost = solv.value(self.running_costs[i])
            print(f"Step {i}: q1 = {state_q1:.4f}, v1 = {state_v1:.4f}, q2 = {state_q2:.4f}, v2 = {state_v2:.4f}, Running cost = {running_cost:.4f}")

        # Terminal cost (only if config.TC == 1)
        if conf.TC:
            terminal_q1 = solv.value(q1[self.N])
            terminal_v1 = solv.value(v1[self.N])
            terminal_q2 = solv.value(q2[self.N])
            terminal_v2 = solv.value(v2[self.N])
            terminal_cost_value = solv.value(self.terminal_cost)
            print(f"Terminal state: q1 = {terminal_q1:.4f}, v1 = {terminal_v1:.4f}, q2 = {terminal_q2:.4f}, v2 = {terminal_v2:.4f}")
            print(f"Terminal cost from NN: {terminal_cost_value:.4f}, calculated at state {[solv.value(q1[self.N]), solv.value(v1[self.N]), solv.value(q2[self.N]), solv.value(v2[self.N])]}")

            # Calculate the true value of the terminal cost
            self.true_terminal_cost_value = self.calculate_true_terminal_cost([terminal_q1, terminal_v1, terminal_q2, terminal_v2])
            self.tc_error = terminal_cost_value - self.true_terminal_cost_value

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
    # Instance of MCP solver
    mpc = MpcDoublePendulum()

    constr_status = "constrained" if conf.costraint else "unconstrained"
    print("==========================")
    print(f"You're running the following test: target: q1 = {conf.q1_target}, q2 = {conf.q2_target}"
          f"initial state: {conf.initial_states}, {constr_status} problem, TC = {conf.TC}, N = {conf.N}, T = {conf.T}")

    num_tests = conf.num_tests
    initial_states = conf.initial_states

    # Create main directory for plots
    output_dir = 'Plots_&_Animations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start timer
    start_time = time.time()

    for test_idx, initial_state in enumerate(initial_states):
        print(f"\n========== Test {test_idx+1}/{num_tests} ==========")
        print(f"Initial state: q1 = {initial_state[0]:.4f}, v1 = {initial_state[1]:.4f}, q2 = {initial_state[2]:.4f}, v2 = {initial_state[3]:.4f}")

        # Create directory for each test
        suffix = conf.nn.split('_DP_')[-1].replace('.h5', '')
        test_dir = os.path.join(output_dir, f'Test_{suffix}_test_{test_idx+1}')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        actual_trajectory = []      # Buffer to store actual trajectory
        actual_inputs = []          # Buffer to store actual inputs
        total_costs = []            # Buffer to store total costs
        terminal_costs = []         # Buffer to store terminal costs
        terminal_costs = []         # Buffer to store terminal costs
        true_tc = []                # Buffer to store true terminal costs
        terminal_cost_error = []    # Buffer to store tc error prediction

        mpc_step = conf.mpc_step
        new_state_guess = np.zeros((4, mpc.N+1))
        new_input_guess = np.zeros((2, mpc.N))

        # First run
        sol = mpc.solve(initial_state)
        actual_trajectory.append(np.array([sol.value(mpc.q1[0]), sol.value(mpc.v1[0]), sol.value(mpc.q2[0]), sol.value(mpc.v2[0])]))
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

            actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
            actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
            total_costs.append(sol.value(mpc.cost))

            if (conf.TC):
                terminal_costs.append(sol.value(mpc.terminal_cost))
                true_tc.append(sol.value(mpc.true_terminal_cost_value))
                terminal_cost_error.append(sol.value(mpc.tc_error))

            # Update state_guess for the next iteration
            for k in range(mpc.N):
                new_state_guess[0, k] = sol.value(mpc.q1[k+1])
                new_state_guess[1, k] = sol.value(mpc.v1[k+1])
                new_state_guess[2, k] = sol.value(mpc.q2[k+1])
                new_state_guess[3, k] = sol.value(mpc.v2[k+1])
            
            # Update input_guess for the next iteration
            for j in range(mpc.N-1):
                new_input_guess[0, j] = sol.value(mpc.u1[j+1])
                new_input_guess[1, j] = sol.value(mpc.u2[j+1])
            print("Step", i+1, "out of", mpc_step, "done")
            if (conf.TC):
                print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_costs[0]), "Terminal Cost", sol.value(mpc.terminal_cost))
            else: 
                print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_costs[0]))
        
        # Stop timer
        end_time = time.time()       

        # Time in nice format
        tot_time = end_time - start_time
        hours = int(tot_time / 3600)
        minutes = int((tot_time - 3600*hours) / 60)       
        seconds = tot_time - hours*3600 - minutes*60
        print("Total elapsed time: {}h {}m {:.2f}s".format(hours, minutes, seconds))

        # Initialize empty lists
        positions_q1 = []
        velocities_v1 = []
        positions_q2 = []
        velocities_v2 = []

        # Extract positions and velocities from the actual trajectory
        for i, state in enumerate(actual_trajectory):
            positions_q1.append(actual_trajectory[i][0])
            velocities_v1.append(actual_trajectory[i][1])
            positions_q2.append(actual_trajectory[i][2])
            velocities_v2.append(actual_trajectory[i][3])

        input1 = [u[0] if u is not None else None for u in actual_inputs]
        input2 = [u[1] if u is not None else None for u in actual_inputs]

        # Plot and save results for the current test
        test_suffix = f'{suffix}_test_{test_idx+1}'

        # Plot and save results
        #plot_and_save(positions_q1, 'mpc step', 'q1 [rad]', 'Position_q1', os.path.join(output_dir, 'position-q1_plot.png'))
        #plot_and_save(velocities_v1, 'mpc step', 'v1 [rad/s]', 'Velocity_q1', os.path.join(output_dir, 'velocity-v1_plot.png'))
        #plot_and_save(positions_q2, 'mpc step', 'q2 [rad]', 'Position_q2', os.path.join(output_dir, 'position-q2_plot.png'))
        #plot_and_save(velocities_v2, 'mpc step', 'v2 [rad/s]', 'Velocity_q2', os.path.join(output_dir, 'velocity-v2_plot.png'))
        #plot_and_save(input1, 'mpc step', 'u [N/m]', 'Torque_u1', os.path.join(output_dir, 'torque_plot-u1.png'))
        #plot_and_save(input2, 'mpc step', 'u [N/m]', 'Torque_u2', os.path.join(output_dir, 'torque_plot-u2.png'))
        #plot_and_save(total_costs, 'MPC Step', 'Total Cost', 'Total Cost', os.path.join(output_dir, 'total_cost_plot.png'))
        #plot_and_save(terminal_costs, 'MPC Step', 'Terminal Cost', 'Terminal Cost', os.path.join(output_dir, 'terminal_cost_plot.png'))

        # Save data in a .csv file
        if conf.TC:
            filename = f'{test_dir}/mpc_DP_TC_{test_suffix}.csv'
            mpc.save_results(positions_q1, velocities_v1, positions_q2, velocities_v2, actual_inputs, total_costs, terminal_costs, true_tc, terminal_cost_error, filename)
        else:
            filename = f'{test_dir}/mpc_DP_NTC_{test_suffix}.csv'
            mpc.save_results(positions_q1, velocities_v1, positions_q2, velocities_v2, actual_inputs, total_costs, terminal_costs, true_tc, terminal_cost_error, filename)
