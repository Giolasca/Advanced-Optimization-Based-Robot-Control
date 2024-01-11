import numpy as np
import casadi
import single_pendulum_dynamics as single_pendulum_dynamics
import multiprocessing
import F_ocp_single_pendulum_conf as conf
import matplotlib.pyplot as plt
import pandas as pd
import time

class OcpSinglePendulum:

    def __init__(self):
        self.T = conf.T                  # OCP horizon
        self.dt = conf.dt                # time step
        self.w_q = conf.w_q              # Position weight
        self.w_u = conf.w_u              # Input weight
        self.w_v = conf.w_v              # Velocity weight
    
    def solve(self, x_init, X_guess = None, U_guess = None):
        self.N = int(self.T/self.dt)                # I initalize the Opti helper from casadi
        self.opti = casadi.Opti()                   # N is the size of the vector we want to realize, the number of steps we want to compute. We create a vector containing all the states of size N+1, 
                                                    # We create a vector of control inputs of size N, one less than the list of states since final control input doesn't have any importance
        # Casadi variables declaration
        self.q = self.opti.variable(self.N+1)       
        self.v = self.opti.variable(self.N+1)
        self.u = self.opti.variable(self.N)
        q = self.q
        v = self.v
        u = self.u
        
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
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = single_pendulum_dynamics.f(np.array([q[i], v[i]]), u[i])
            # Dynamics imposition
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])
        
        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit, q[i], conf.upperPositionLimit))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit, v[i], conf.upperVelocityLimit))
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound, u[i], conf.upperControlBound))
        
        return self.opti.solve()


if __name__ == "__main__":
    
    # Instance of OCP solver
    ocp = OcpSinglePendulum()

    # Generate an array of states either randomly or in a gridded pattern
    if (conf.grid == 1):
        npos = 121
        nvel = 121
        n_ics = npos * nvel
        state_array = conf.grid_states(npos, nvel)
    else:
        nrandom = 100
        state_array = conf.random_states(nrandom)


    # Function definition to run in a process
    def ocp_function_single_pendulum(index):
        # Empy lists to store viable and non viable states
        viable = []
        no_viable = []

        # We divide the states grid in complementary subsets
        for i in range(index[0], index[1]):
            x = state_array[i, :]
            try:
                sol = ocp.solve(x)
                viable.append([x[0], x[1]])
                print("Feasible initial state found: [{:.3f}   {:.3f}]".format(*x))
            except RuntimeError as e:                     # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found: [{:.3f}   {:.3f}]".format(*x))
                    no_viable.append([x[0], x[1]])
                else:
                    print("Runtime error:", e)
        return viable, no_viable


    # Multi process execution
    if (conf.multiproc == 1):
        print("Multiprocessing execution started, number of processes:", conf.num_processes)
        
        # Subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, n_ics, num=conf.num_processes+1)

        # I define the arguments to pass to the functions: the indexes necessary to split the states grid
        args = []
        for i in range(conf.num_processes):
            args.append([int(indexes[i]), int(indexes[i+1])])

        # I initiate the pool
        pool = multiprocessing.Pool(processes=conf.num_processes)

        # Function to keep track of execution time
        start = time.time()

        # Multiprocess start
        results = pool.map(ocp_function_single_pendulum, args)

        # Multiprocess end
        pool.close()
        pool.join()

        # Stop keeping track of time
        end = time.time()

        # Time in nice format
        tot_time = end-start
        seconds = int(tot_time % 60)
        minutes = int((tot_time - seconds) / 60)       
        hours = int((tot_time - seconds - minutes*60) / 3600)
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

        # Regroup the results into 2 lists of viable and non viable states
        viable_states = np.array(results[0][0])
        no_viable_states = np.array(results[0][1])
        for i in range(conf.num_processes-1):
            viable_states = np.concatenate((viable_states, np.array(results[i+1][0])))
            no_viable_states = np.concatenate((no_viable_states, np.array(results[i+1][1])))


    # Single process execution
    else:               
        print("Single process execution started")

        # Empty lists to store viable and non viable states
        viable_states = []
        no_viable_states = []

        # Keep track of execution time
        start = time.time()
        # Iterate through every state in the states grid
        for state in state_array:
            try:
                sol = ocp.solve(state)
                viable_states.append([state[0], state[1]])
                print("Feasible initial state found:", state)
            except RuntimeError as e:      # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", state)
                    no_viable_states.append([state[0], state[1]])
                else:
                    print("Runtime error:", e)

        # Stop keeping track of time
        end = time.time()

        viable_states = np.array(viable_states)
        no_viable_states = np.array(no_viable_states)

        # Execution time in a nice format
        tot_time = end-start
        seconds = tot_time % 60
        minutes = (tot_time - seconds) / 60        
        hours = (tot_time - seconds - minutes*60) / 3600
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

    # Plot the overall viable and non viable states
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:,0], viable_states[:,1], c='r', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

    # Unify both viable and non viable states with a flag to show wether they're viable or not
    viable_states = np.column_stack((viable_states, np.ones(len(viable_states), dtype=int)))
    no_viable_states = np.column_stack((no_viable_states, np.zeros(len(no_viable_states), dtype=int)))
    dataset = np.concatenate((viable_states, no_viable_states))

    # Create a DataFrame starting from the final array
    columns = ['q', 'v', 'viable']
    df = pd.DataFrame(dataset, columns=columns)

    # Export DataFrame to csv format
    df.to_csv('data_single.csv', index=False)