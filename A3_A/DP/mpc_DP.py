import numpy as np
import casadi
import DP_dynamics as DP_dynamics
import mpc_DP_conf as conf
from nn import create_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class MpcDoublePendulum:

    def __init__(self):
        self.N = conf.N                     # Number of steps
        self.w_q = conf.w_q                 # Position weight
        self.w_u = conf.w_u                 # Input weight
        self.w_v = conf.w_v                 # Velocity weight

        self.model = create_model(4)        # Template of NN for double pendulum
        self.model.load_weights(conf.nn)
        self.weights = self.model.get_weights()
        self.mean, self.std = conf.init_scaler()
    
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

        state = [(q1[self.N] - self.mean[0])/self.std[0], (v1[self.N] - self.mean[1])/self.std[1], 
                 (q2[self.N] - self.mean[2])/self.std[2], (v2[self.N] - self.mean[3])/self.std[3]]
        
        # Cost definition
        self.cost = 0
        self.terminal_cost = self.nn_to_casadi(self.weights, state)
        self.running_costs = [None] * (self.N+1)
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * (v1[i]**2 + v2[i]**2)
            self.running_costs[i] += self.w_q * ((q1[i] - q1_target)**2 + (q2[i] - q2_target)**2)
            if (i<self.N):   
                self.running_costs[i] += self.w_u * (u1[i]**2 + u2[i]**2)
            self.cost += self.running_costs[i] 

        # Adding terminal cost from the neural network
        if (conf.TC == 1):
            self.cost +=  self.terminal_cost    
            self.opti.minimize(self.cost)
        else:
            self.cost = self.cost
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
        sol = self.opti.solve()

        '''
        # Print execution costs and terminal costs   
        solv = self.opti.solve()
        for i in range(self.N+1):
            running_cost_value = float(solv.value(self.running_costs[i]))
            print(f"Running Cost at step {i}: {running_cost_value:.6f}")
        state_print = [solv.value(self.q1)[self.N], solv.value(self.v1)[self.N], solv.value(self.q2)[self.N], solv.value(self.v2)[self.N]]
        print("State:", state_print)
        state_print_normalized = [(solv.value(self.q1)[self.N-1]-self.mean[0])/self.std[0], (solv.value(self.v1)[self.N-1]-self.mean[1])/self.std[1], 
                                  (solv.value(self.q2)[self.N-1]-self.mean[2])/self.std[2], (solv.value(self.v2)[self.N-1]-self.mean[3])/self.std[3]]
        print("Normalized state:", state_print_normalized)
        terminal_cost_value = float(solv.value(self.terminal_cost))
        print(f"Terminal Cost: {terminal_cost_value:.6f}")
        total_cost_value = float(solv.value(self.cost))
        print(f"Total Cost: {total_cost_value:.6f}")
        '''

        return sol

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

    initial_state = conf.initial_state
    actual_trajectory = []      # Buffer to store actual trajectory
    actual_inputs = []          # Buffer to store actual inputs
    total_costs = []            # Buffer to store total costs
    terminal_costs = []         # Buffer to store terminal costs
    
    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    # Start timer
    start_time = time.time()

    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
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
        terminal_costs.append(sol.value(mpc.terminal_cost))

        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q1[k+1])
            new_state_guess[1, k] = sol.value(mpc.v1[k+1])
            new_state_guess[2, k] = sol.value(mpc.q2[k+1])
            new_state_guess[3, k] = sol.value(mpc.v2[k+1])
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

    positions_q1 = []
    velocities_v1 = []
    positions_q2 = []
    velocities_v2 = []

    for i, state in enumerate(actual_trajectory):
        positions_q1.append(actual_trajectory[i][0])
        velocities_v1.append(actual_trajectory[i][1])
        positions_q2.append(actual_trajectory[i][2])
        velocities_v2.append(actual_trajectory[i][3])

    input1 = [u[0] if u is not None else None for u in actual_inputs]
    input2 = [u[1] if u is not None else None for u in actual_inputs]

    output_dir = 'Plots_&_Animations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot and save results
    plot_and_save(positions_q1, 'mpc step', 'q1 [rad]', 'Position_q1', os.path.join(output_dir, 'position-q1_plot.png.png'))
    plot_and_save(velocities_v1, 'mpc step', 'v1 [rad/s]', 'Velocity_q1', os.path.join(output_dir, 'velocity-v1_plot.png'))
    plot_and_save(positions_q2, 'mpc step', 'q2 [rad]', 'Position_q2', os.path.join(output_dir, 'position-q2_plot.png.png'))
    plot_and_save(velocities_v2, 'mpc step', 'v2 [rad/s]', 'Velocity_q2', os.path.join(output_dir, 'velocity-v2_plot.png'))
    plot_and_save(input1, 'mpc step', 'u [N/m]', 'Torque_u1', os.path.join(output_dir, 'torque_plot-u1.png'))
    plot_and_save(input2, 'mpc step', 'u [N/m]', 'Torque_u2', os.path.join(output_dir, 'torque_plot-u2.png'))
    plot_and_save(total_costs, 'MPC Step', 'Total Cost', 'Total Cost', os.path.join(output_dir, 'total_cost_plot.png'))
    plot_and_save(terminal_costs, 'MPC Step', 'Terminal Cost', 'Terminal Cost', os.path.join(output_dir, 'terminal_cost_plot.png'))

    # Save data in a .csv file
    filename = 'Plots_&_Animations/MPC_DoublePendulum_TC.csv'
    mpc.save_results(positions_q1, velocities_v1, positions_q2, velocities_v2, actual_inputs, total_costs, terminal_costs, filename)
    