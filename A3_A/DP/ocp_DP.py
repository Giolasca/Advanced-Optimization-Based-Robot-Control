import numpy as np
import casadi
import DP_dynamics
import multiprocessing
import ocp_DP_conf as config
import pandas as pd
import os
import time

# Class defining the optimal control problem for a double pendulum
class OcpDoublePendulum:

    def __init__(self):
        self.N = config.N                    # OCP horizon
        self.w_q1 = config.w_q1              # Position weight
        self.w_u1 = config.w_u1              # Input weight
        self.w_v1 = config.w_v1              # Velocity weight

        self.w_q2 = config.w_q2              # Position weight
        self.w_u2 = config.w_u2              # Input weight
        self.w_v2 = config.w_v2              # Velocity weight

    def save_results(self, state_buffer):        # Save results in a csv file to create the DataSet
        filename = 'ocp_data_DP.csv'
        positions_q1 = [state[0] for state in state_buffer]
        velocities_v1 = [state[1] for state in state_buffer]
        positions_q2 = [state[2] for state in state_buffer]
        velocities_v2 = [state[3] for state in state_buffer]
        costs = [state[4] for state in state_buffer]
        df = pd.DataFrame({'q1': positions_q1, 'v1': velocities_v1, 'q2': positions_q2, 'v2': velocities_v2, 'Costs': costs})
        df.to_csv(filename, index=False)

    def solve(self, initial_state, state_guess=None, control_guess=None):
        self.opti = casadi.Opti()
        
        # Declaration of the variables in casaDi types
        self.q1 = self.opti.variable(self.N + 1)       
        self.v1 = self.opti.variable(self.N + 1)
        self.u1 = self.opti.variable(self.N)
        q1, v1, u1 = self.q1, self.v1, self.u1

        self.q2 = self.opti.variable(self.N + 1)       
        self.v2 = self.opti.variable(self.N + 1)
        self.u2 = self.opti.variable(self.N)
        q2, v2, u2 = self.q2, self.v2, self.u2

        # State vector initialization
        for i in range(self.N + 1):
            self.opti.set_initial(q1[i], state_guess[0, i] if state_guess is not None else initial_state[0])
            self.opti.set_initial(v1[i], state_guess[1, i] if state_guess is not None else initial_state[1])
            self.opti.set_initial(q2[i], state_guess[2, i] if state_guess is not None else initial_state[2])
            self.opti.set_initial(v2[i], state_guess[3, i] if state_guess is not None else initial_state[3])

        # Control input vector initalization
        if control_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(u1[i], control_guess[0, i])
                self.opti.set_initial(u2[i], control_guess[1, i])
        
        # Cost definition
        self.cost = 0
        for i in range(self.N + 1):
            self.cost += self.w_v1 * self.v1[i]**2 + self.w_v2 * self.v2[i]**2
            if i < self.N:
                self.cost += self.w_u1 * self.u1[i]**2 + self.w_u2 * self.u2[i]**2
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = DP_dynamics.f(np.array([self.q1[i], self.v1[i], self.q2[i], self.v2[i]]), np.array([self.u1[i], self.u2[i]]))
            # Dynamics imposition
            self.opti.subject_to(self.q1[i + 1] == x_next[0])
            self.opti.subject_to(self.v1[i + 1] == x_next[1])
            self.opti.subject_to(self.q2[i + 1] == x_next[2])
            self.opti.subject_to(self.v2[i + 1] == x_next[3])

        # Initial state constraint
        self.opti.subject_to(self.q1[0] == initial_state[0])
        self.opti.subject_to(self.v1[0] == initial_state[1])
        self.opti.subject_to(self.q2[0] == initial_state[2])
        self.opti.subject_to(self.v2[0] == initial_state[3])

        '''
        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(config.q1_min, q1[i], config.q1_max))
            self.opti.subject_to(self.opti.bounded(config.q2_min, q2[i], config.q2_max))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(config.v1_min, v1[i], config.v1_max))
            self.opti.subject_to(self.opti.bounded(config.v2_min, v2[i], config.v2_max))

        # Control bounds
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(config.u1_min, u1[i], config.u1_max))
            self.opti.subject_to(self.opti.bounded(config.u2_min, u2[i], config.u2_max))
        '''

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opts = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opts)
        
        return self.opti.solve()

# Function to execute the optimal control problem in parallel
def ocp_task(index, state_array, ocp):
    state_buffer = []       # Buffer to store initial states
    cost_buffer = []        # Buffer to store optimal costs

    for i in range(index[0], index[1]):
        x = state_array[i, :]
        try:
            sol = ocp.solve(x)
            cost_buffer.append(sol.value(ocp.cost))
            print("State: [{:.4f}  {:.4f}   {:.4f}   {:.4f}] Cost {:.4f}".format(*x, cost_buffer[-1]))
            state_buffer.append([x[0], x[1], x[2], x[3], cost_buffer[-1]])
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("Could not solve for: [{:.4f}   {:.4f}   {:.4f}   {:.4f}]".format(*x))
            else:
                print("Runtime error:", e)
    return state_buffer

if __name__ == "__main__":
    # Initialize the optimal control problem
    ocp = OcpDoublePendulum()

        # Generate an array of states either randomly or in a gridded pattern
    if (config.grid == 1):
        n_pos_q1 = 21
        n_vel_v1 = 21   
        n_pos_q2 = 21
        n_vel_v2 = 21
        state_array = config.grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2)
    else:
        num_pairs = 10  
        state_array = config.random_states(num_pairs)

    # Multi process execution
    if config.multiproc == 1:
        print("Multiprocessing execution started, number of processes:", config.num_processes)
        print("Total points: {}  Calculated points: {}".format(config.tot_points, config.end_index))

        # Subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, state_array.shape[0], num=config.num_processes + 1)
        args = [[int(indexes[i]), int(indexes[i + 1])] for i in range(config.num_processes)]
        
        with multiprocessing.Pool(processes=config.num_processes) as pool:
            start_time = time.time()
            results = pool.starmap(ocp_task, [(arg, state_array, ocp) for arg in args])
            end_time = time.time()

        # Store the results
        x0_costs = np.concatenate(results)

    else:
        print("Single process execution")

        # Full range of indices for single process
        index_range = (0, len(state_array))

        # Start execution time
        start_time = time.time()

        # Process all initial states in a single call to ocp_task
        state_buffer = ocp_task(index_range, state_array, ocp)
        
        # End execution time
        end_time = time.time()

        # Store the results
        x0_costs = np.concatenate(state_buffer)

    # Time in nice format
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

    # Save data in a .csv file
    ocp.save_results(x0_costs)