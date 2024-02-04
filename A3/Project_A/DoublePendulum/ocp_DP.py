import numpy as np 
import casadi
import DP_dynamics as DP_dynamics 
import multiprocessing
import ocp_DP_conf as conf 
import matplotlib.pyplot as plt
import pandas as pd
import time

class OcpSinglePendulum:

    def __init__(self):
        self.N = conf.N              # OCP horizon
        self.w_v1 = conf.w_v1          # Velocity weight link 1
        self.w_u1 = conf.w_u1          # Input weight link 1

        self.w_v2 = conf.w_v2          # Velocity weight link 2
        self.w_u2 = conf.w_u2          # Input weight link 2

        self.q1_min = conf.q1_min          # Lower position limit link 1
        self.q1_max = conf.q1_max          # Upper position limit link 1
        self.v1_min = conf.v1_min          # Lower velocity limit link 1
        self.v1_max = conf.v1_max          # Upper velocity limit link 1
        self.u1_min = conf.u1_min          # Lower control bound link 1
        self.u1_max = conf.u1_max          # Upper control bound link 1

        self.q2_min = conf.q2_min          # Lower position limit link 2
        self.q2_max = conf.q2_max          # Upper position limit link 2
        self.v2_min = conf.v2_min          # Lower velocity limit link 2
        self.v2_max = conf.v2_max          # Upper velocity limit link 2
        self.u2_min = conf.u2_min          # Lower control bound link 2
        self.u2_max = conf.u2_max          # Upper control bound link 2

        self.buffer_cost = []       # Buffer to store optimal costs

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()       # Initialize Casadi variables

        # Create vectors for states and control inputs
        # Size of state vector is N+1, as it includes initial and final states
        # Size of control input is N, since the final control input is not significant
        self.q1 = self.opti.variable(N+1)    # states
        self.v1 = self.opti.variable(N+1)    # velocities
        self.u1 = self.opti.variable(N)      # control inputs

        # Alias variables for convenience
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1

        # Create vectors for states and control inputs
        # Size of state vector is N+1, as it includes initial and final states
        # Size of control input is N, since the final control input is not significant
        self.q2 = self.opti.variable(N+1)    # states
        self.v2 = self.opti.variable(N+1)    # velocities
        self.u2 = self.opti.variable(N)      # control inputs

        # Alias variables for convenience
        q2 = self.q2
        v2 = self.v2
        u2 = self.u2

        # State vector initialization
        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(self.q1[i], X_guess[0,i])
                self.opti.set_initial(self.v1[i], X_guess[1,i])
                self.opti.set_initial(self.q2[i], X_guess[2,i])
                self.opti.set_initial(self.v2[i], X_guess[3,i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(self.q1[i], x_init[0])
                self.opti.set_initial(self.v1[i], x_init[1])
                self.opti.set_initial(self.q2[i], x_init[2])
                self.opti.set_initial(self.v2[i], x_init[3])
        
        # Control input vector initalization
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess[0,i])
                self.opti.set_initial(u2[i], U_guess[1,i])

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            if (i < N):
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

        # State bound constraints
        for i in range(N+1):
            self.opti.subject_to(self.opti.bounded(self.q1_min, q1[i], self.q1_max))
            self.opti.subject_to(self.opti.bounded(self.v1_min, v1[i], self.v1_max))
            self.opti.subject_to(self.opti.bounded(self.q2_min, q2[i], self.q2_max))
            self.opti.subject_to(self.opti.bounded(self.v2_min, v2[i], self.v2_max))

            #  Control bounds Constraints
            if (i<self.N):
                for i in range(N):
                    self.opti.subject_to(self.opti.bounded(self.u1_min, u1[i], self.u1_max))
                    self.opti.subject_to(self.opti.bounded(self.u2_min, u2[i], self.u2_max))


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
        n_pos_q1 = 3
        n_vel_v1 = 6   # 2 - 3 - 6 - 11 - 21 
        n_pos_q2 = 21
        n_vel_v2 = 21
        n_ics = n_pos_q1 * n_vel_v1 * n_pos_q2 * n_vel_v2
        state_array = conf.grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2)
    else:
        num_pairs = 10  
        state_array = conf.random_states(num_pairs)
    
    def ocp_single_pendulum(index):
        states = []         # Buffer to store initial states

        for i in range(index[0], index[1]):
            state = state_array[i, :]
            try:
                sol = ocp.solve(state, conf.N)
                print("State: [{:.3f}  {:.3f}  {:.3f}  {:.3f}] Cost {:.3f}".format(*state, ocp.buffer_cost[-1]))
                states.append([state[0],state[1], state[2], state[3], ocp.buffer_cost[-1]])
            except Exception as e:
                print("Could not solve OCP for [{:.3f}  {:.3f}  {:.3f}  {:.3f}]".format(*state))
        return states

    # Multi process execution
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
    print("Total elapsed time: {}h {}min {:.2f}s".format(hours, minutes, seconds))

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
    df.to_csv('ocp_data_SP.csv', index=False)


# Plotting the cost over the initial states
plt.scatter(np.array(x0_costs)[:, 0], np.array(x0_costs)[:, 1], c=x0_costs[:, 2], cmap='viridis')
plt.xlabel('Initial Position (q)')
plt.ylabel('Initial Velocity (v)')
plt.title('Cost over Initial States')
plt.colorbar(label='Cost')
plt.show()