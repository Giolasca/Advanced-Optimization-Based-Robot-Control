import numpy as np
import casadi
import DP_dynamics as dynamics
import mpc_DP_conf as config
from nn_DP import create_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class MpcDoublePendulum:

    def __init__(self):
        self.N = config.N                       # OCP horizon
        self.w_q1 = config.w_q1                 # weight on first link position
        self.w_u1 = config.w_u1                 # weight on first link control
        self.w_v1 = config.w_v1                 # weight on first link velocity  

        self.w_q2 = config.w_q2                 # weight on second link position
        self.w_u2 = config.w_u2                 # weight on second link control
        self.w_v2 = config.w_v2                 # weight on second link velocity 

        self.model = create_model(4)        # Template of NN
        self.model.load_weights(config.nn)
        self.weights = self.model.get_weights()
        self.mean_x, self.std_x, self.mean_y, self.std_y = config.init_scaler() 

    def save_results(self, pos1, vel1, pos2, vel2, actual_inputs, total_costs, terminal_costs, filename):
        max_length = max(len(pos1), len(vel1), len(pos2), len(vel2), len(actual_inputs), len(total_costs), len(terminal_costs))
        pos1.extend([None] * (max_length - len(pos1)))
        vel1.extend([None] * (max_length - len(vel1)))
        pos2.extend([None] * (max_length - len(pos2)))
        vel2.extend([None] * (max_length - len(vel2)))
        actual_inputs.extend([None] * (max_length - len(actual_inputs)))
        total_costs.extend([None] * (max_length - len(total_costs)))
        terminal_costs.extend([None] * (max_length - len(terminal_costs)))

        df = pd.DataFrame({
            'q1': pos1,
            'v1': vel1,
            'q2': pos2,
            'v2': vel2,
            'u1': [u[0] if u is not None else None for u in actual_inputs],
            'u2': [u[1] if u is not None else None for u in actual_inputs],
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

            it += 1

        return casadi.MX(out[0])

    
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
        q1_des, q2_des = config.q1_target, config.q2_target

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

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)

        # Terminal cost (if enabled by config.TC)
        state = [(q1[self.N] - self.mean_x[0])/self.std_x[0], (v1[self.N] - self.mean_x[1])/self.std_x[1], (q2[self.N] - self.mean_x[2])/self.std_x[2], (v2[self.N] - self.mean_x[3])/self.std_x[3]]
        #state = [q[self.N], v[self.N]]
        self.terminal_cost = self.nn_to_casadi(self.weights, state)
        
        # Cost functional
        self.cost = 0
        self.running_cost = [None,]*(self.N+1)
        for i in range(self.N + 1):
            self.running_cost[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            self.running_cost[i] += self.w_q1 * (q1[i] - q1_des)**2 + self.w_q2 * (q2[i] - q2_des)**2
            if (i<self.N):
                self.running_cost[i] += self.w_u1 * u1[i]**2 + self.w_u2 * u2[i]**2
            self.cost += self.running_cost[i]
        self.opti.minimize(self.cost)

        # Adding terminal cost from the neural network (if config.TC == 1)
        if config.TC:
            self.cost += 1e-2*self.terminal_cost

        # Minimize the total cost
        self.opti.minimize(self.cost)

         # Dynamics constraint
        for i in range(self.N):
            x_next = dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(v1[i+1] == x_next[1])
            self.opti.subject_to(q2[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])

        # Initial state constraint       
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])
        
        if config.costraint:
            # State bound constraints
            for i in range(self.N+1):
                self.opti.subject_to(self.opti.bounded(config.q1_min, q1[i], config.q1_max))
                self.opti.subject_to(self.opti.bounded(config.v1_min, v1[i], config.v1_max))
                self.opti.subject_to(self.opti.bounded(config.q2_min, q2[i], config.q2_max))
                self.opti.subject_to(self.opti.bounded(config.v1_min, v2[i], config.v2_max))
            for i in range(self.N):
                self.opti.subject_to(self.opti.bounded(config.u1_min, u1[i], config.u1_max))
                self.opti.subject_to(self.opti.bounded(config.u2_min, u2[i], config.u2_max))
            
        # Solve the optimization problem
        solv = self.opti.solve()

        # === Debug window ===
        current_state = np.array([solv.value(q1[0]), solv.value(v1[0]), solv.value(q2[0]), solv.value(v2[0])])  # Current state
        rescaled_state = np.array([(current_state[0] - self.mean_x[0]) / self.std_x[0], (current_state[1] - self.mean_x[1]) / self.std_x[1], (current_state[2] - self.mean_x[2]) / self.std_x[2], (current_state[3] - self.mean_x[3]) / self.std_x[3]])  # Rescaled state

        print("==== MPC Step Debug Info ====")
        print(f"Current state: q1 = {current_state[0]:.4f}, v1 = {current_state[1]:.4f}, q2 = {current_state[1]:.4f}, v2 = {current_state[2]:.4f}")
        print(f"Rescaled state: q1' = {rescaled_state[0]:.4f}, v1' = {rescaled_state[1]:.4f}, q2' = {rescaled_state[0]:.4f}, v2' = {rescaled_state[1]:.4f}")

        # Display running costs and states
        print(f"Running costs for each step:")
        for i in range(self.N+1):
            state_q1 = solv.value(q1[i])
            state_v1 = solv.value(v1[i])
            state_q2 = solv.value(q2[i])
            state_v2 = solv.value(v2[i])
            running_cost = solv.value(self.running_cost[i])
            print(f"Step {i}: q1 = {state_q1:.4f}, v1 = {state_v1:.4f}, q2 = {state_q2:.4f}, v2 = {state_v2:.4f}, Running cost = {running_cost:.4f}")

        # Terminal cost (only if config.TC == 1)
        if config.TC:
            terminal_q1 = solv.value(q1[self.N])
            terminal_v1 = solv.value(v1[self.N])
            terminal_q2 = solv.value(q2[self.N])
            terminal_v2 = solv.value(v2[self.N])
            terminal_cost_value = solv.value(self.terminal_cost)
            print(f"Terminal state: q1 = {terminal_q1:.4f}, v1 = {terminal_v1:.4f}, q2 = {terminal_q2:.4f}, v2 = {terminal_v2:.4f}")
            print(f"Terminal cost from NN: {terminal_cost_value:.4f}, calculated at state {[solv.value(q1[self.N]), solv.value(v1[self.N]), solv.value(q2[self.N]), solv.value(v2[self.N])]}")

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
    mpc = MpcDoublePendulum()

    constr_status = "constrained" if config.costraint else "unconstrained"
    print("==========================")
    print(f"You're running the following test: target: q1 = {config.q1_target}, q2 = {config.q2_target}"
          f"initial state: {config.initial_state}, {constr_status} problem, TC = {config.TC}, N = {config.N}, T = {config.T}")
    
    initial_state = config.initial_state     # Initial state of the pendulum (position and velocity)
    actual_trajectory = []      # Buffer to store actual trajectory
    actual_inputs = []          # Buffer to store actual inputs
    total_costs = []            # Buffer to store total costs
    terminal_costs = []         # Buffer to store terminal costs

    mpc_step = config.mpc_step          # Number of MPC steps
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    # Start timer
    start_time = time.time()
    
    # First run
    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[0]), sol.value(mpc.v1[0]), sol.value(mpc.q2[0]), sol.value(mpc.v2[0])]))
    actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))

    # Creation of state_guess of size 4 x N+1
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q1[i+1])
        new_state_guess[1, i] = sol.value(mpc.v1[i+1])
        new_state_guess[2, i] = sol.value(mpc.q2[i+1])
        new_state_guess[3, i] = sol.value(mpc.v2[i+1])

    # Creation of input_guess of size 2 x N
    for i in range(mpc.N-1):
        new_input_guess[0, i] = sol.value(mpc.u1[i+1])
        new_input_guess[1, i] = sol.value(mpc.u2[i+1])
    
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
        
        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
        total_costs.append(sol.value(mpc.cost))
        terminal_costs.append(sol.value(mpc.terminal_cost))

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

        if (config.TC):
            print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_cost[0]), "Terminal Cost", sol.value(mpc.terminal_cost))
        else: 
            print("Cost", sol.value(mpc.cost), "Running cost", sol.value(mpc.running_cost[0]))
        
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

    # Create directory for plots
    output_dir = 'Plots_&_Animations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the suffix from the file name
    suffix = config.nn.split('_DP_')[-1].replace('.h5', '')

    # Create dedicate directory for the test
    test_dir = os.path.join(output_dir, f'Test_{suffix}')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Plot and save results
    plot_and_save(positions_q1, 'mpc step', 'q [rad]', 'q1', os.path.join(test_dir, f'position_plot_q1_{suffix}.png'))
    plot_and_save(positions_q2, 'mpc step', 'q [rad]', 'q2', os.path.join(test_dir, f'position_plot_q2_{suffix}.png'))
    plot_and_save(velocities_v1, 'mpc step', 'v [rad/s]', 'v1', os.path.join(test_dir, f'velocity_plot_v1_{suffix}.png'))
    plot_and_save(velocities_v2, 'mpc step', 'v [rad/s]', 'v2', os.path.join(test_dir, f'velocity_plot_v2_{suffix}.png'))
    plot_and_save(input1, 'mpc step', 'u [N/m]', 'Torque', os.path.join(test_dir, f'torque_plot_u1_{suffix}.png'))
    plot_and_save(input2, 'mpc step', 'u [N/m]', 'Torque', os.path.join(test_dir, f'torque_plot_u2_{suffix}.png'))
    plot_and_save(total_costs, 'MPC Step', 'Total Cost', 'Total Cost', os.path.join(test_dir, f'total_cost_plot_{suffix}.png'))
    plot_and_save(terminal_costs, 'MPC Step', 'Terminal Cost', 'Terminal Cost', os.path.join(test_dir, f'terminal_cost_plot_{suffix}.png'))

    # Save data in a .csv file
    if (config.TC):
        mpc.save_results(positions_q1, positions_q1, velocities_v1, velocities_v1, input1, input2, total_costs, terminal_costs, f'Plots_&_Animations/mpc_SP_TC_{suffix}.csv')
    else:
        if(config.scenario_type):
            mpc.save_results(positions_q1, positions_q1, velocities_v1, velocities_v1, input1, input2, total_costs, terminal_costs, f'Plots_&_Animations/mpc_SP_NTC_T_1_{suffix}.csv')
        else:
            mpc.save_results(positions_q1, positions_q1, velocities_v1, velocities_v1, input1, input2, total_costs, terminal_costs, f'Plots_&_Animations/mpc_SP_NTC_T_0.01_{suffix}.csv')