import numpy as np
import casadi
import doublependulum_dynamics as doublependulum_dynamics
import multiprocessing
import ocp_double_pendulum_conf as conf

class OcpDoublePendulum:

    def __init__(self):
        self.T = conf.T                  # OCP horizon
        self.dt = conf.dt                # time step
        self.w_q1 = conf.w_q1             # Position weight
        self.w_u1 = conf.w_u1              # Input weight
        self.w_v1 = conf.w_v1              # Velocity weight

        self.w_q2 = conf.w_q2             # Position weight
        self.w_u2 = conf.w_u2              # Input weight
        self.w_v2 = conf.w_v2              # Velocity weight
    
    def solve(self, x_init1, x_init2, X_guess1 = None, X_guess2 = None, U_guess1 = None, U_guess2 = None):
        self.N = int(self.T/self.dt)                # I initalize the Opti helper from casadi
        self.opti = casadi.Opti()                   # N is the size of the vector we want to realize, the number of steps we want to compute. We create a vector containing all the states of size N+1, 
                                                    # We create a vector of control inputs of size N, one less than the list of states since final control input doesn't have any importance
        # Casadi variables declaration
        self.q1 = self.opti.variable(self.N+1)
        self.q2 = self.opti.variable(self.N+1)       
        self.v1 = self.opti.variable(self.N+1)
        self.v2 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        self.u2 = self.opti.variable(self.N)
        q1 = self.q1
        q2 = self.q2
        v1 = self.v1
        v2 = self.v2
        u1 = self.u1
        u2 = self.u2
        
        # State vector initialization
        if ((X_guess1 is not None) and (X_guess2 is not None)):
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess1[0,i])
                self.opti.set_initial(v1[i], X_guess1[1,i])
                self.opti.set_initial(q2[i], X_guess2[0,i])
                self.opti.set_initial(v2[i], X_guess2[1,i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], x_init1[0])
                self.opti.set_initial(v1[i], x_init1[1])
                self.opti.set_initial(q2[i], x_init2[0])
                self.opti.set_initial(v2[i], x_init2[1])
        
        # Control input vector initalization
        if ((U_guess1 is not None) and (U_guess2 is not None)):
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess1[i])
                self.opti.set_initial(u2[i], U_guess2[i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Cost definition
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
            x_next = doublependulum_dynamics.f(np.array([q1[i], q2[i], v1[i], v2[i]]), np.array([u1[i], u2[i]]))
            # Dynamics imposition
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(q2[i+1] == x_next[1])
            self.opti.subject_to(v1[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])
        
        # Initial state constraint
        self.opti.subject_to(q1[0] == x_init1[0])
        self.opti.subject_to(v1[0] == x_init1[1])
        self.opti.subject_to(q2[0] == x_init2[0])
        self.opti.subject_to(v2[0] == x_init2[1])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(q1[i] <= conf.upperPositionLimit1)
            self.opti.subject_to(q1[i] >= conf.lowerPositionLimit1)
            self.opti.subject_to(q2[i] <= conf.upperPositionLimit2)
            self.opti.subject_to(q2[i] >= conf.lowerPositionLimit2)

            # Velocity bounds
            self.opti.subject_to(v1[i] <= conf.upperVelocityLimit1)
            self.opti.subject_to(v1[i] >= conf.lowerVelocityLimit1)
            self.opti.subject_to(v2[i] <= conf.upperVelocityLimit2)
            self.opti.subject_to(v2[i] >= conf.lowerVelocityLimit2)
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(u1[i] <= conf.upperControlBound1)
                self.opti.subject_to(u1[i] >= conf.lowerControlBound1)
                self.opti.subject_to(u2[i] <= conf.upperControlBound2)
                self.opti.subject_to(u2[i] >= conf.lowerControlBound2)

        return self.opti.solve()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    multiproc = conf.multiproc
    num_processes = conf.num_processes

    # Creation of initial states grid
    n_pos1 = 10
    n_vel1 = 10
    n_ics1 = n_pos1 * n_vel1

    n_pos2 = 10
    n_vel2 = 10
    n_ics2 = n_pos2 * n_vel2

    possible_q1 = np.linspace(conf.lowerPositionLimit1, conf.upperPositionLimit1, num=n_pos1)
    possible_v1 = np.linspace(conf.lowerVelocityLimit1, conf.upperVelocityLimit1, num=n_vel1)
    possible_q2 = np.linspace(conf.lowerPositionLimit2, conf.upperPositionLimit2, num=n_pos2)
    possible_v2 = np.linspace(conf.lowerVelocityLimit2, conf.upperVelocityLimit2, num=n_vel2)
    state_array1 = np.zeros((n_ics1, 2))
    state_array2 = np.zeros((n_ics2, 2))
 
    #state_array = np.hstack((state_array1,state_array2))

    j = k = 0
    for i in range (n_ics1):
        state_array1[i,:] = np.array([possible_q1[j], possible_v1[k]])
        state_array2[i,:] = np.array([possible_q2[j], possible_v2[k]])
        k += 1
        if (k == n_vel1):
            k = 0
            j += 1
    # Instance of OCP solver
    ocp_double_pendulum = OcpDoublePendulum()

    # Function definition to run in a process
    def ocp_function_double_pendulum(index):
        # I create empy lists to store viable and non viable states
        viable = []
        no_viable = []
        # We divide the states grid in complementary subsets
        for i in range(index[0], index[1]):
            x1 = state_array1[i, :]
            x2 = state_array2[i, :]
            try:
                sol = ocp_double_pendulum.solve(x1,x2)
                viable.append([x1[0], x1[1], x2[0], x2[1]])
                print("Feasible initial state found:", x1,x2)
            except RuntimeError as e:                     # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", x1,x2)
                    no_viable.append([x1[0], x1[1], x2[0], x2[1]])
                else:
                    print("Runtime error:", e)
        return viable, no_viable

    if (multiproc == 1):
        # I subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, n_ics1, num=num_processes+1)

        # I define the arguments to pass to the functions: the indexes necessary to split the states grid
        args = []
        for i in range(num_processes):
            args.append([int(indexes[i]), int(indexes[i+1])])

        # I initiate the pool
        pool = multiprocessing.Pool(processes=num_processes)

        # Function to keep track of execution time
        start = time.time()

        # Multiprocess start
        results = pool.map(ocp_function_double_pendulum, args)

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
        for state1 in state_array1:
            try:
                sol = ocp_double_pendulum.solve(state1)
                viable_states.append([state1[0], state1[1]])
                print("Feasible initial state found:", state1)
            except RuntimeError as e:      # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", state1)
                    no_viable_states.append([state1[0], state1[1]])
                else:
                    print("Runtime error:", e)
        
        for state2 in state_array2:
            try:
                sol = ocp_double_pendulum.solve(state2)
                viable_states.append([state2[0], state2[1]])
                print("Feasible initial state found:", state2)
            except RuntimeError as e:      # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", state2)
                    no_viable_states.append([state2[0], state2[1]])
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
        ax.scatter(viable_states[:,2], viable_states[:,3], c='r', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,2], no_viable_states[:,3], c='b', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()