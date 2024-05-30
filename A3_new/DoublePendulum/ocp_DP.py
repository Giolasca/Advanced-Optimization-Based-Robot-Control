import numpy as np
import casadi
import DP_dynamics 
import multiprocessing
import ocp_DP_conf as conf
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

class OcpDoublePendulum:

    def __init__(self):
        self.T = conf.T                  # OCP horizon
        self.dt = conf.dt                # time step
        self.w_q1 = conf.w_q1              # Position weight
        self.w_u1 = conf.w_u1              # Input weight
        self.w_v1 = conf.w_v1              # Velocity weight

        self.w_q2 = conf.w_q2              # Position weight
        self.w_u2 = conf.w_u2              # Input weight
        self.w_v2 = conf.w_v2              # Velocity weight


    def solve(self, x_init, X_guess = None, U_guess = None):
        self.N = int(self.T/self.dt)                # I initalize the Opti helper from casadi
        self.opti = casadi.Opti()                   # N is the size of the vector we want to realize, the number of steps we want to compute. We create a vector containing all the states of size N+1, 
                                                    # We create a vector of control inputs of size N, one less than the list of states since final control input doesn't have any importance
        # Casadi variables declaration
        self.q1 = self.opti.variable(self.N+1)       
        self.v1 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1

        self.q2 = self.opti.variable(self.N+1)       
        self.v2 = self.opti.variable(self.N+1)
        self.u2 = self.opti.variable(self.N)
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

        # Cost definition
        i = 0
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u1 * u1[i]**2 + self.w_u2 * u2[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = DP_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
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
            self.opti.subject_to(self.opti.bounded(conf.q1_min, q1[i], conf.q1_max))
            self.opti.subject_to(self.opti.bounded(conf.q2_min, q2[i], conf.q2_max))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.v1_min, v1[i], conf.v1_max))
            self.opti.subject_to(self.opti.bounded(conf.v2_min, v2[i], conf.v2_max))

        for i in range(self.N):
            # Control bounds
            self.opti.subject_to(self.opti.bounded(conf.u1_min, u1[i], conf.u1_max))
            self.opti.subject_to(self.opti.bounded(conf.u2_min, u2[i], conf.u2_max))


        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        return self.opti.solve()


if __name__ == "__main__":
    
    # Instance of OCP solver
    ocp = OcpDoublePendulum()
    
    # Generate an array of states either randomly or in a gridded pattern
    if (conf.grid == 1):
        n_pos_q1 = 21
        n_vel_v1 = 21   
        n_pos_q2 = 21
        n_vel_v2 = 21
        state_array = conf.grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2)
    else:
        num_pairs = 10  
        state_array = conf.random_states(num_pairs)


    def ocp_function_double_pendulum(index):
        # Empy lists to store state and costs
        states = []
        costs = []

        # We divide the states grid in complementary subsets
        for i in range(index[0], index[1]):
            x = state_array[i, :]
            try:
                sol = ocp.solve(x)
                costs.append(sol.value(ocp.cost))
                print("State: [{:.4f}  {:.4f}   {:.4f}   {:.4f}] Cost {:.4f}".format(*x, costs[-1]))
                states.append([x[0], x[1], x[2], x[3], costs[-1]])
            except RuntimeError as e:        # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Could not solve for: [{:.4f}   {:.4f}   {:.4f}   {:.4f}]".format(*x))
                else:
                    print("Runtime error:", e)
        return states

    # Multi process execution
    points = conf.end_index - conf.start_index
    print("Multiprocessing execution started, number of processes: ", conf.num_processes)
    print("Total points: {}  Calculated points: {}".format(conf.tot_points, conf.end_index))

    # Subdivide the states grid in equal spaces proportional to the number of processes
    indexes = np.linspace(0, state_array.shape[0], num=conf.num_processes+1)

    # I define the arguments to pass to the functions: the indexes necessary to split the states grid
    args = []
    for i in range(conf.num_processes):
        args.append([int(indexes[i]), int(indexes[i+1])])

    # I initiate the pool
    pool = multiprocessing.Pool(processes=conf.num_processes)

    # Function to keep track of execution time
    start = time.time()

    # Multiprocess start
    results = pool.map(ocp_function_double_pendulum, args)

    # Multiprocess end
    pool.close()
    pool.join()
        
    # Stop keeping track of time
    end = time.time()

    # Time in nice format
    tot_time = end-start
    hours = int(tot_time / 3600)
    minutes = int((tot_time - 3600*hours) / 60)       
    seconds = tot_time - hours*3600 - minutes*60
    print("Total elapsed time: {}h {}min {:.2f}s".format(hours, minutes, seconds))

    # Regroup the results into 2 lists of viable and non viable states
    x0_costs = np.array(results[0])  

    for i in range(conf.num_processes - 1):
        x0_costs = np.concatenate((x0_costs, np.array(results[i + 1])))

    # Create a DataFrame starting from the final array
    df = pd.DataFrame({'q1': x0_costs[:, 0], 'v1': x0_costs[:, 1], 'q2': x0_costs[:, 2], 'v2': x0_costs[:, 3],  'Costs': x0_costs[:, 4]})

    # Salva il DataFrame in un file CSV in modalità append
    df.to_csv('ocp_data_8.csv', mode='a', index=False, header=not os.path.exists('ocp_data_8.csv'))