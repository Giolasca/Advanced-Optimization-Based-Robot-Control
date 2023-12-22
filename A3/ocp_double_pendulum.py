import numpy as np
import casadi
import doublependulum_dynamics as doublependulum_dynamics
import multiprocessing
import ocp_double_pendulum_conf as conf

class OcpDoublePendulum:

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
            self.running_costs[i] = self.w_v * v[i]*2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = doublependulum_dynamics.f(np.array([q[i], v[i]]), u[i])
            # Dynamics imposition
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])
        
        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(q[i] <= conf.upperPositionLimit)
            self.opti.subject_to(q[i] >= conf.lowerPositionLimit)

            # Velocity bounds
            self.opti.subject_to(v[i] <= conf.upperVelocityLimit)
            self.opti.subject_to(v[i] >= conf.lowerVelocityLimit)
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(u[i] <= conf.upperControlBound)
                self.opti.subject_to(u[i] >= conf.lowerControlBound)

        return self.opti.solve()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    multiproc = conf.multiproc
    num_processes = conf.num_processes

    # Creation of initial states grid
    n_pos = 101
    n_vel = 101
    n_ics = n_pos * n_vel
    possible_q = np.linspace(conf.lowerPositionLimit, conf.upperPositionLimit, num=n_pos)
    possible_v = np.linspace(conf.lowerVelocityLimit, conf.upperVelocityLimit, num=n_vel)
    state_array = np.zeros((n_ics, 2))

    j = k = 0
    for i in range (n_ics):
        state_array[i,:] = np.array([possible_q[j], possible_v[k]])
        k += 1
        if (k == n_vel):
            k = 0
            j += 1

    # Instance of OCP solver
    ocp = OcpDoublePendulum()

    # Function definition to run in a process
    def ocp_function(index):
        # I create empy lists to store viable and non viable states
        viable = []
        no_viable = []
        # We divide the states grid in complementary subsets
        for i in range(index[0], index[1]):
            x = state_array[i, :]
            try:
                sol = ocp.solve(x)
                viable.append([x[0], x[1]])
                print("Feasible initial state found:", x)
            except RuntimeError as e:                     # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", x)
                    no_viable.append([x[0], x[1]])
                else:
                    print("Runtime error:", e)
        return viable, no_viable


    if (multiproc == 1):
        # I subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, n_ics, num=num_processes+1)

        # I define the arguments to pass to the functions: the indexes necessary to split the states grid
        args = []
        for i in range(num_processes):
            args.append([int(indexes[i]), int(indexes[i+1])])

        # I initiate the pool
        pool = multiprocessing.Pool(processes=num_processes)

        # Function to keep track of execution time
        start = time.time()

        # Multiprocess start
        results = pool.map(ocp_function, args)

        # Multiprocess end
        pool.close()
        pool.join()
        
        # I regroup the results into 2 lists of viable and non viable states
        viable_states = np.array(results[0][0])
        no_viable_states = np.array(results[0][1])
        for i in range(num_processes-1):
            viable_states = np.concatenate((viable_states, np.array(results[i+1][0])))
            no_viable_states = np.concatenate((no_viable_states, np.array(results[i+1][1])))

        # Stop keeping track of time
        end = time.time()

        # Time in nice format
        tot_time = end-start
        seconds = int(tot_time % 60)
        minutes = int((tot_time - seconds) / 60)       
        hours = int((tot_time - seconds - minutes*60) / 3600)
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")


    else:               # Single process execution

        # I create empty lists to store viable and non viable states
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