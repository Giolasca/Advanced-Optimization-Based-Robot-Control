import numpy as np
import casadi 
import doublependulum_dynamics as double_pendulum_dynamics
import multiprocessing
import ocp_double_pendulum_conf as conf

class OcpDoublePendulum:

    def __init__(self):
        self.T = conf.T             # OCP horizon
        self.dt = conf.dt           # time step
        # self.w_q1 = conf.w_q1       # Weights       
        self.w_v1 = conf.w_v1       
        self.w_u1 = conf.w_u1       
        # self.w_q2 = conf.w_q2       
        self.w_v2 = conf.w_v2       
        self.w_u2 = conf.w_u2       

    def solve(self, x_init, X_guess = None, U_guess = None):
        self.N = int(self.T/self.dt)
        self.opti = casadi.Opti()

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

        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess[0, i])
                self.opti.set_initial(v1[i], X_guess[1, i])
                self.opti.set_initial(q2[i], X_guess[2, i])
                self.opti.set_initial(v2[i], X_guess[3, i])
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess[0, i])
                self.opti.set_initial(u2[i], U_guess[1, i])

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
            x_next = double_pendulum_dynamics.f(np.array([q1[i], q2[i], v1[i], v2[i]]), np.array([u1[i], u2[i]]))
            # Dynamics imposition
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(v1[i+1] == x_next[2])
            self.opti.subject_to(q2[i+1] == x_next[1])
            self.opti.subject_to(v2[i+1] == x_next[3])
        
        # Initial state constraint
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit1, q1[i], conf.upperPositionLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit2, q2[i], conf.upperPositionLimit2))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit1, v1[i], conf.upperVelocityLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit2, v2[i], conf.upperVelocityLimit2))
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound1, u1[i], conf.upperControlBound1))
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound2, u2[i], conf.upperControlBound2))
        
        return self.opti.solve()

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    multiproc = conf.multiproc
    num_processes = conf.num_processes

    n_ics, state_array = conf.grid_states(11, 11, 11, 11)
    # n_ics, state_array = conf.random_states(100)

    ocp = OcpDoublePendulum()

    def ocp_function(index):
        viable = []
        no_viable = []
        for i in range(index[0], index[1]):
            x = state_array[i, :]
            try:
                sol = ocp.solve(x)
                viable.append([x[0], x[1], x[2], x[3]])
                print("Feasible initial state found:", x)
            except RuntimeError as e:
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", x)
                    no_viable.append([x[0], x[1], x[2], x[3]])
                else:
                    print("Runtime error:", e)

        return viable, no_viable

    if (multiproc == 1):
        print("Multiprocessing execution started, number of processes:", num_processes)
        indexes = np.linspace(0, n_ics, num=num_processes+1)

        args = []
        for i in range(num_processes):
            args.append([int(indexes[i]), int(indexes[i+1])])

        pool = multiprocessing.Pool(processes=num_processes)

        start = time.time()

        results = pool.map(ocp_function, args)

        pool.close()
        pool.join()

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

        print("Single process execution started")

        # I create empty lists to store viable and non viable states
        viable_states = []
        no_viable_states = []

        # Keep track of execution time
        start = time.time()
        # Iterate through every state in the states grid
        for state in state_array:
            try:
                sol = ocp.solve(state)
                viable_states.append([state[0], state[1], state[2], state[3]])
                print("Feasible initial state found:", state)
            except RuntimeError as e:      # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", state)
                    no_viable_states.append([state[0], state[1], state[2], state[3]])
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
    columns = ['q1', 'v1', 'q2', 'v2', 'viable']
    df = pd.DataFrame(dataset, columns=columns)

    # Export DataFrame to csv format
    df.to_csv('double_data.csv', index=False)