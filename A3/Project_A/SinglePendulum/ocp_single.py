import numpy as np
import casadi
import SP_dynamics 
import multiprocessing
import ocp_SP_conf as conf
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

    def save_to_csv(self, x_0_buffer, cost_buffer, filename):
        # Estrai q e v dai buffer
        q_values = [state[0] for state in x_0_buffer]
        v_values = [state[1] for state in x_0_buffer]

        # Crea un DataFrame
        df = pd.DataFrame({'q': q_values, 'v': v_values, 'cost': cost_buffer})

        # Salva il DataFrame in un file CSV
        df.to_csv(filename, index=False)

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
            x_next = SP_dynamics.f(np.array([q[i], v[i]]), u[i])
            # Dynamics imposition
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])
        
        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(conf.q_min, q[i], conf.q_max))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.v_min, v[i], conf.v_max))
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(self.opti.bounded(conf.u_min, u[i], conf.u_max))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

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
        x_0_buffer = []
        cost_buffer = []
        # We divide the states grid in complementary subsets
        for i in range(index[0], index[1]):
            state = state_array[i, :]
            try:
                sol = ocp.solve(state)
                x_0_buffer.append([state[0], state[1]])
                cost_buffer.append(sol.value(ocp.cost))
                print("State: [{:.3f}  {:.3f}] Cost {:.3f}".format(*state, cost_buffer[-1]))
            except RuntimeError as e:                     # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Could not solve for: [{:.3f}   {:.3f}]".format(*state))
                else:
                    print("Runtime error:", e)
        return x_0_buffer, cost_buffer


    # Multi process execution
    if (conf.multiproc == 1):
        print("Multiprocessing execution started, number of processes:", conf.num_processes)
        
        # Subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, n_ics, num=conf.num_processes+1)

        # I define the arguments to pass to the functions: the indexes necessary to split the states grid
        args = []
        for i in range(conf.num_processes):
            args.append((int(indexes[i]), int(indexes[i+1])))

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

        # Combine results from multiprocessing
        x_0_buffer_combined = []
        cost_buffer_combined = []
        for result in results:
            x_0_buffer_combined.extend(result[0])
            cost_buffer_combined.extend(result[1])

    # Single process execution
    else:
        print("Single process execution started")

        # Empty lists to store viable and non viable states
        x_0_buffer = []
        cost_buffer = []

        # Keep track of execution time
        start = time.time()
        # Iterate through every state in the states grid
        for state in state_array:
            try:
                sol = ocp.solve(state)
                x_0_buffer.append([state[0], state[1]])
                cost_buffer.append(sol.value(ocp.cost))
                print("Feasible initial state found:", state, "Cost:", sol.value(ocp.cost))
            except RuntimeError as e:
                if "Infeasible_Problem_Detected" in str(e):
                    print("Could not solve for:", state)
                else:
                    print("Runtime error:", e)

    # Stop keeping track of time
    end = time.time()

    # Execution time in a nice format
    tot_time = end - start
    seconds = tot_time % 60
    minutes = (tot_time - seconds) / 60
    hours = (tot_time - seconds - minutes * 60) / 3600
    print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

    # Salva i dati in un file CSV
    ocp.save_to_csv(x_0_buffer_combined, cost_buffer_combined, 'ocp_data.csv')

    # Plotting the cost over the initial states
    plt.scatter(np.array(x_0_buffer_combined)[:, 0], np.array(x_0_buffer_combined)[:, 1], c=cost_buffer, cmap='viridis')
    plt.xlabel('Initial Position (q)')
    plt.ylabel('Initial Velocity (v)')
    plt.title('Cost over Initial States')
    plt.colorbar(label='Cost')
    plt.show()