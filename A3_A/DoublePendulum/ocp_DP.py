import numpy as np
import casadi
import DP_dynamics
import multiprocessing
import ocp_DP_conf as conf
import pandas as pd
import os
import time

# Class defining the optimal control problem for a double pendulum
class OcpDoublePendulum:
    def __init__(self):
        # Load configuration parameters
        self.T = conf.T
        self.dt = conf.dt
        self.w_q1 = conf.w_q1
        self.w_u1 = conf.w_u1
        self.w_v1 = conf.w_v1
        self.w_q2 = conf.w_q2
        self.w_u2 = conf.w_u2
        self.w_v2 = conf.w_v2

    # Initialize variables for the optimization problem
    def initialize_variables(self, X_guess, U_guess, x_init):
        self.q1 = self.opti.variable(self.N + 1)
        self.v1 = self.opti.variable(self.N + 1)
        self.u1 = self.opti.variable(self.N)
        self.q2 = self.opti.variable(self.N + 1)
        self.v2 = self.opti.variable(self.N + 1)
        self.u2 = self.opti.variable(self.N)

        q1, v1, q2, v2 = self.q1, self.v1, self.q2, self.v2

        # Set initial guesses for state variables
        for i in range(self.N + 1):
            self.opti.set_initial(q1[i], X_guess[0, i] if X_guess is not None else x_init[0])
            self.opti.set_initial(v1[i], X_guess[1, i] if X_guess is not None else x_init[1])
            self.opti.set_initial(q2[i], X_guess[2, i] if X_guess is not None else x_init[2])
            self.opti.set_initial(v2[i], X_guess[3, i] if X_guess is not None else x_init[3])

        # Set initial guesses for control variables
        if U_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(self.u1[i], U_guess[0, i])
                self.opti.set_initial(self.u2[i], U_guess[1, i])

    # Set up the optimization problem with constraints and cost function
    def setup_optimization_problem(self, x_init):
        self.cost = 0
        # Define the cost function
        for i in range(self.N + 1):
            self.cost += self.w_v1 * self.v1[i]**2 + self.w_v2 * self.v2[i]**2
            if i < self.N:
                self.cost += self.w_u1 * self.u1[i]**2 + self.w_u2 * self.u2[i]**2
        self.opti.minimize(self.cost)

        # Define the dynamic constraints using the system dynamics
        for i in range(self.N):
            x_next = DP_dynamics.f(np.array([self.q1[i], self.v1[i], self.q2[i], self.v2[i]]), np.array([self.u1[i], self.u2[i]]))
            self.opti.subject_to(self.q1[i + 1] == x_next[0])
            self.opti.subject_to(self.v1[i + 1] == x_next[1])
            self.opti.subject_to(self.q2[i + 1] == x_next[2])
            self.opti.subject_to(self.v2[i + 1] == x_next[3])

        # Define the initial conditions
        self.opti.subject_to(self.q1[0] == x_init[0])
        self.opti.subject_to(self.v1[0] == x_init[1])
        self.opti.subject_to(self.q2[0] == x_init[2])
        self.opti.subject_to(self.v2[0] == x_init[3])

        # Define the bounds on state and control variables
        '''
        for i in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(conf.q1_min, self.q1[i], conf.q1_max))
            self.opti.subject_to(self.opti.bounded(conf.q2_min, self.q2[i], conf.q2_max))
            self.opti.subject_to(self.opti.bounded(conf.v1_min, self.v1[i], conf.v1_max))
            self.opti.subject_to(self.opti.bounded(conf.v2_min, self.v2[i], conf.v2_max))
        
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(conf.u1_min, self.u1[i], conf.u1_max))
            self.opti.subject_to(self.opti.bounded(conf.u2_min, self.u2[i], conf.u2_max))
        '''
        # Solver options for IPOPT
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)

    # Solve the optimization problem
    def solve(self, x_init, X_guess=None, U_guess=None):
        self.N = int(self.T / self.dt)
        self.opti = casadi.Opti()
        
        self.initialize_variables(X_guess, U_guess, x_init)
        self.setup_optimization_problem(x_init)
        
        return self.opti.solve()

# Generate the state array based on the grid or random configuration
def generate_state_array():
    if conf.grid == 1:
        n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2 = 21, 21, 21, 21
        return conf.grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2)
    else:
        num_pairs = 10
        return conf.random_states(num_pairs)

# Function to execute the optimal control problem in parallel
def ocp_function_double_pendulum(index, state_array, ocp):
    states, costs = [], []
    for i in range(index[0], index[1]):
        x = state_array[i, :]
        try:
            sol = ocp.solve(x)
            costs.append(sol.value(ocp.cost))
            print("State: [{:.4f}  {:.4f}   {:.4f}   {:.4f}] Cost {:.4f}".format(*x, costs[-1]))
            states.append([x[0], x[1], x[2], x[3], costs[-1]])
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("Could not solve for: [{:.4f}   {:.4f}   {:.4f}   {:.4f}]".format(*x))
            else:
                print("Runtime error:", e)
    return states

if __name__ == "__main__":
    # Initialize the optimal control problem
    ocp = OcpDoublePendulum()
    state_array = generate_state_array()

    # Function to manage parallel execution
    def parallel_ocp_execution():
        indexes = np.linspace(0, state_array.shape[0], num=conf.num_processes + 1)
        args = [[int(indexes[i]), int(indexes[i + 1])] for i in range(conf.num_processes)]
        
        with multiprocessing.Pool(processes=conf.num_processes) as pool:
            start = time.time()
            results = pool.starmap(ocp_function_double_pendulum, [(arg, state_array, ocp) for arg in args])
            end = time.time()
        
        return results, end - start

    # Start the multiprocessing execution
    print("Multiprocessing execution started, number of processes:", conf.num_processes)
    print("Total points: {}  Calculated points: {}".format(conf.tot_points, conf.end_index))
    results, tot_time = parallel_ocp_execution()

    # Calculate and print total elapsed time
    hours, rem = divmod(tot_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total elapsed time: {int(hours)}h {int(minutes)}min {seconds:.2f}s")

    # Save results to CSV
    x0_costs = np.concatenate(results)
    df = pd.DataFrame(x0_costs, columns=['q1', 'v1', 'q2', 'v2', 'Costs'])
    df.to_csv('ocp_data_8.csv', mode='a', index=False, header=not os.path.exists('ocp_data_8.csv'))
