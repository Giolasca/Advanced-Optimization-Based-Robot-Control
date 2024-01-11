import numpy as np
import casadi
import doublependulum_dynamics as double_pendulum_dynamics
import multiprocessing
import ocp_double_pendulum_conf as conf
import matplotlib.pyplot as plt
import pandas as pd
import time


class OcpDoublePendulum:

    def __init__(self):
        self.T = conf.T                  # OCP horizon
        self.dt = conf.dt                # time step
        self.w_q1 = conf.w_q1              # Position weight link1
        self.w_u1 = conf.w_u1              # Input weight link1
        self.w_v1 = conf.w_v1              # Velocity weight link1
        self.w_q2 = conf.w_q2              # Position weight link2
        self.w_u2 = conf.w_u2              # Input weight link2
        self.w_v2 = conf.w_v2              # Velocity weight link2
    
    def solve(self, x_init, X_guess = None, U_guess = None):
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

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

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
            x_next = double_pendulum_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
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
            self.opti.subject_to(q1[i] <= conf.upperPositionLimit_q1)
            self.opti.subject_to(q1[i] >= conf.lowerPositionLimit_q1)
            self.opti.subject_to(q2[i] <= conf.upperPositionLimit_q2)
            self.opti.subject_to(q2[i] >= conf.lowerPositionLimit_q2)

            # Velocity bounds
            self.opti.subject_to(v1[i] <= conf.upperVelocityLimit_v1)
            self.opti.subject_to(v1[i] >= conf.lowerVelocityLimit_v1)
            self.opti.subject_to(v2[i] <= conf.upperVelocityLimit_v2)
            self.opti.subject_to(v2[i] >= conf.lowerVelocityLimit_v2)
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(u1[i] <= conf.upperControlBound_u1)
                self.opti.subject_to(u1[i] >= conf.lowerControlBound_u1)
                self.opti.subject_to(u2[i] <= conf.upperControlBound_u2)
                self.opti.subject_to(u2[i] >= conf.lowerControlBound_u2)

        return self.opti.solve()


if __name__ == "__main__":

    # Instance of OCP solver
    ocp = OcpDoublePendulum()


    # Generate an array of states either randomly or in a gridded pattern
    if (conf.grid == 1):
        n_pos_q1 = 6
        n_vel_v1 = 6   # 2 - 3 - 6 - 11 - 21 
        n_pos_q2 = 21
        n_vel_v2 = 21
        n_ics = n_pos_q1 * n_vel_v1 * n_pos_q2 * n_vel_v2
        state_array = conf.grid_states(n_pos_q1, n_vel_v1, n_pos_q2, n_vel_v2)
    else:
        num_pairs = 10  
        state_array = conf.random_states(num_pairs)


    # Load existing points from a dataset to simplify the task
    if (conf.old_data == 1):
    
        # Load data from CSV
        data = pd.read_csv("double_data.csv")

        # Filter data based on the 'viable' column
        viable_states_old = data[data['viable_states'] == 1][['q1', 'v1', 'q2', 'v2']].values
        non_viable_states_old = data[data['viable_states'] == 0][['q1', 'v1', 'q2', 'v2']].values

        # Convert arrays to lists of tuples
        viable_tuples = [tuple(x) for x in viable_states_old]
        non_viable_tuples = [tuple(x) for x in non_viable_states_old]

        # Create separate sets for viable and non-viable states
        viable_set = set(viable_tuples)
        non_viable_set = set(non_viable_tuples)

        # Combine the two sets to get all existing points
        all_existing_points = viable_set.union(non_viable_set)

        # New_state_array 
        new_state_array = np.array([point for point in state_array if tuple(point) not in all_existing_points])

    
    # Function definition to run in a process
    def ocp_function_double_pendulum(index):
        # Empy lists to store viable and non viable states
        viable = []
        no_viable = []

        # We divide the states grid in complementary subsets
        for i in range(index[0], index[1]):
            
            if (conf.old_data == 1):
                x = new_state_array[i, :]
            else: 
                x = state_array[i, :]
        
            try:
                sol = ocp.solve(x)
                viable.append([x[0], x[1], x[2], x[3]])
                print("Feasible initial state found:", x)
            except RuntimeError as e:                     # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", x)
                    no_viable.append([x[0], x[1], x[2], x[3]])
                else:
                    print("Runtime error:", e)
        return viable, no_viable


    # Multi process execution
    if (conf.multiproc == 1):
        print("Multiprocessing execution started, number of processes:", conf.num_processes, "number of points", new_state_array.shape)
        
        # Subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, new_state_array.shape[0], num=conf.num_processes+1)

        # I define the arguments to pass to the functions: the indexes necessary to split the states grid
        args = []
        i = 0
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
        seconds = int(tot_time % 60)
        minutes = int((tot_time - seconds) / 60)       
        hours = int((tot_time - seconds - minutes*60) / 3600)
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

        # Regroup the results into 2 lists of viable and non-viable states
        viable_states = np.array(results[0][0])
        non_viable_states = np.array(results[0][1])

        for i in range(conf.num_processes - 1):

            # Concatenate viable_states
            viable_states_temp = np.array(results[i + 1][0])
            if viable_states_temp.size > 0:  # Check that the vector is not empty
                viable_states = np.concatenate((viable_states, viable_states_temp))

            # Concatenate no_viable_states
            non_viable_states_temp = np.array(results[i + 1][1])
            if non_viable_states_temp.size > 0:  # Check that the vector is not empty
                non_viable_states = np.concatenate((non_viable_states, non_viable_states_temp))


        if(conf.save_data == 1):
            
            # Check if viable_states is not empty before concatenating
            if viable_states.size == 0:
                viable_states_new = viable_states_old if viable_states_old.size != 0 else []
            else:
                viable_states_new = viable_states if viable_states_old.size == 0 else np.concatenate((viable_states_old, viable_states))

            # Check if no_viable_states is not empty before concatenating
            if non_viable_states.size == 0:
                non_viable_states_new = non_viable_states_old if non_viable_states_old.size != 0 else []
            else:
                non_viable_states_new = non_viable_states if non_viable_states_old.size == 0 else np.concatenate((non_viable_states_old, non_viable_states))

            # Unify both viable and non viable states with a flag to show whether they're viable or not
            viable_states_combined = np.column_stack((viable_states_new, np.ones(len(viable_states_new), dtype=int)))
            non_viable_states_combined = np.column_stack((non_viable_states_new, np.zeros(len(non_viable_states_new), dtype=int)))
            dataset_combined = np.concatenate((viable_states_combined, non_viable_states_combined))

            # Create a DataFrame starting from the final array
            columns_combined = ['q1', 'v1', 'q2', 'v2', 'viable_states']
            df_combined = pd.DataFrame(dataset_combined, columns=columns_combined)

            # Export DataFrame to CSV format
            df_combined.to_csv('double_data.csv', index=False)

            
    # Single process execution
    else:               

        # I create empty lists to store viable and non viable states
        viable = []
        no_viable = []

        # Keep track of execution time
        start = time.time()
        state = state_array[0,:]
        # Iterate through every state in the states grid
        for state in state_array:
            try:
                sol = ocp.solve(state)
                viable.append([state[0], state[1], state[2], state[3]])
                print("Feasible initial state found:", state)
            except RuntimeError as e:      # We catch the runtime exception relative to absence of solution
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found:", state)
                    no_viable.append([state[0], state[1], state[2], state[3]])
                else:
                    print("Runtime error:", e)

        # Stop keeping track of time
        end = time.time()

        viable_states = np.array(viable)
        non_viable_states_new = np.array(no_viable)

        # Execution time in a nice format
        tot_time = end-start
        seconds = tot_time % 60
        minutes = (tot_time - seconds) / 60        
        hours = (tot_time - seconds - minutes*60) / 3600
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")