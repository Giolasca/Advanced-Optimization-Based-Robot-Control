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
        self.N = conf.N              # OCP horizon
        self.w_v = conf.w_v          # Velocity weight
        self.w_u = conf.w_u          # Input weight
        self.q_min = conf.q_min          # Lower position limit
        self.q_max = conf.q_max          # Upper position limit
        self.v_min = conf.v_min          # Lower velocity limit
        self.v_max = conf.v_max          # Upper velocity limit
        self.u_min = conf.u_min          # Lower control bound
        self.u_max = conf.u_max          # Upper control bound

        self.buffer_cost = []       # Buffer to store optimal costs
        self.buffer_x0 = []         # Buffer to store initial states

    def save_to_csv(self, filename='SP_costs.csv'):
        data = {'Position (q)': np.array(self.buffer_x0)[:, 0],
                'Velocity (v)': np.array(self.buffer_x0)[:, 1],
                'Cost': self.buffer_cost}
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()       # Initialize Casadi variables

        # Create vectors for states and control inputs
        # Size of state vector is N+1, as it includes initial and final states
        # Size of control input is N, since the final control input is not significant
        self.q = self.opti.variable(N+1)    # states
        self.v = self.opti.variable(N+1)    # velocities
        self.u = self.opti.variable(N)      # control inputs

        # Alias variables for convenience
        q = self.q
        v = self.v
        u = self.u

        # State Vector initialization
        if X_guess is not None:
            for i in range(N+1):
                self.opti.set_initial(self.q, X_guess[:, 0])
                self.opti.set_initial(self.v, X_guess[:, 1])
        else:
            for i in range(N+1):
                self.opti.set_initial(self.q, x_init[0])
                self.opti.set_initial(self.v, x_init[1])

        # Control input initialization
        if U_guess is not None:
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i, :])

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
            self.opti.subject_to(self.opti.bounded(self.q_min, q[i], self.q_max))
            self.opti.subject_to(self.opti.bounded(self.v_min, v[i], self.v_max))

            #  Control bounds Constraints
            if (i<self.N):
                for i in range(N):
                    self.opti.subject_to(self.opti.bounded(self.u_min, u[i], self.u_max))

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()

        # Store initial state and optimal cost in buffers
        self.buffer_x0.append([x_init[0], x_init[1]])
        self.buffer_cost.append(sol.value(self.cost))

        return sol


if __name__ == "__main__":

    # Instance of OCP solver
    ocp = OcpSinglePendulum()

    # Generate an array of states either randomly or in a gridded pattern
    if (conf.grid == 1):
        npos = 10
        nvel = 10
        n_ics = npos * nvel
        state_array = conf.grid_states(npos, nvel)
    else:
        nrandom = 100
        state_array = conf.random_states(nrandom)

    def ocp_single_pendulum(index):
        for i in range(index[0], index[1]):
            state = state_array[i, :]
            try:
                sol = ocp.solve(state, conf.N)
                print("State: [{:.3f}  {:.3f}] Cost {:.3f}".format(*state, ocp.buffer_cost[-1]))
            except Exception as e:
                print("Could not solve OCP for [{:.3f}  {:.3f}]".format(*state))

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

    # Multiprocess star
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

    ocp.save_to_csv()


# Plotting the cost over the initial states
plt.scatter(np.array(ocp.buffer_x0)[:, 0], np.array(ocp.buffer_x0)[:, 1], c=ocp.buffer_cost, cmap='viridis')
plt.xlabel('Initial Position (q)')
plt.ylabel('Initial Velocity (v)')
plt.title('Cost over Initial States')
plt.colorbar(label='Cost')
plt.show()