import numpy as np 
import casadi
import SP_dynamics as SP_dynamics 
import multiprocessing
import ocp_SP_conf as conf 
import matplotlib.pyplot as plt
import pandas as pd
import time

class OcpSinglePendulum:

    def __init__(self):
        self.N = conf.N             # OCP horizon
        self.w_v = conf.w_v         # Velocity weight
        self.w_u = conf.w_u         # Input weight

        self.buffer_cost = []       # Buffer to store optimal costs

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()       # Initialize Casadi variables

        # Create vectors for states and control inputs
        # Size of state vector is N+1, as it includes initial and final states
        # Size of control input is N, since the final control input is not significant
        self.q = self.opti.variable(N+1)    # States
        self.v = self.opti.variable(N+1)    # Velocities
        self.u = self.opti.variable(N)      # Control inputs

        # Alias variables for convenience
        q, v, u = self.q, self.v, self.u

        # State Vector initialization
        initial_q = X_guess[:, 0] if X_guess is not None else x_init[0]
        initial_v = X_guess[:, 1] if X_guess is not None else x_init[1]

        for i in range(N+1):
            self.opti.set_initial(self.q, initial_q)
            self.opti.set_initial(self.v, initial_v)

        # Control input initialization
        initial_u = U_guess if U_guess is not None else np.zeros((N, u.shape[1]))

        for i in range(N):
            self.opti.set_initial(u[i], initial_u[i, :])

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            if (i < N):
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics Constraint
        for i in range(N):
            x_next = SP_dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i+1] == x_next[0])
            self.opti.subject_to(v[i+1] == x_next[1])

        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # State bound constraints
        for i in range(N+1):
            self.opti.subject_to(self.opti.bounded(conf.q_min, q[i], conf.q_max))
            self.opti.subject_to(self.opti.bounded(conf.v_min, v[i], conf.v_max))

        #  Control bounds Constraints
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(conf.u_min, u[i],conf.u_max))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()

        # Store initial state and optimal cost in buffers
        self.buffer_cost.append(sol.value(self.cost))

        return sol


if __name__ == "__main__":

    # Instance of OCP solver
    ocp = OcpSinglePendulum()

    # Generate an array of states either randomly or in a gridded pattern
    if (conf.grid == 1):
        npos = 5
        nvel = 5
        n_ics = npos * nvel
        state_array = conf.grid_states(npos, nvel)
    else:
        nrandom = 100
        state_array = conf.random_states(nrandom)
    
    def ocp_single_pendulum(index):
        states = []         # Buffer to store initial states

        for i in range(index[0], index[1]):
            state = state_array[i, :]
            try:
                sol = ocp.solve(state, conf.N)
                print("State: [{:.3f}  {:.3f}] Cost {:.3f}".format(*state, ocp.buffer_cost[-1]))
                states.append([state[0],state[1], ocp.buffer_cost[-1]])
            except Exception as e:
                print("Could not solve OCP for [{:.3f}  {:.3f}]".format(*state))
        return states

    # Multi process execution
    print("Multiprocessing execution started, number of processes:", conf.num_processes)
        
    # Subdivide the states grid in equal spaces proportional to the number of processes
    indexes = np.linspace(0, n_ics, num=conf.num_processes+1)

    # Define the arguments to pass to the functions: the indexes necessary to split the states grid
    args = []
    for i in range(conf.num_processes):
        args.append((int(indexes[i]), int(indexes[i+1])))

    # Initiate the pool
    pool = multiprocessing.Pool(processes=conf.num_processes)

    # Function to keep track of execution time
    start = time.time()

    # Multiprocess start
    results = pool.map(ocp_single_pendulum, args)

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
    print("Total elapsed time: {}h {}m {:.2f}s".format(hours, minutes, seconds))

    # Regroup the results into 2 lists of viable and non viable states
    x0_costs = np.array(results[0])  

    for i in range(conf.num_processes - 1):
        x0_costs = np.concatenate((x0_costs, np.array(results[i + 1])))

    # Create a DataFrame starting from the final array
    df = pd.DataFrame({
    'Initial Position (q)': x0_costs[:, 0],  
    'Initial Velocity (v)': x0_costs[:, 1],  
    'Costs': x0_costs[:, 2]      
    })

    # Salva il DataFrame in un file CSV
    df.to_csv('ocp_data_SP_1.csv', index=False)

# Plotting the cost over the initial states
plt.scatter(np.array(x0_costs)[:, 0], np.array(x0_costs)[:, 1], c=x0_costs[:, 2], cmap='viridis')
plt.xlabel('Initial Position (q)')
plt.ylabel('Initial Velocity (v)')
plt.title('Cost over Initial States')
plt.colorbar(label='Cost')
plt.show()