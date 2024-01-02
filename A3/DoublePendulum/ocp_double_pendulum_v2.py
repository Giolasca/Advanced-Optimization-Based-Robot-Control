import numpy as np
import casadi
import scipy.io
import doublependulum_dynamics as doublependulum_dynamics
import multiprocessing
import ocp_double_pendulum_conf as conf
import matplotlib.pyplot as plt


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
            x_next = doublependulum_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
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
    import matplotlib.pyplot as plt
    import time

    multiproc = conf.multiproc
    num_processes = conf.num_processes
    plot = conf.plot
    plot1 = conf.plot1

    # Creation of initial states grid
    n_pos = 51
    n_vel = 51
    n_ics = n_pos * n_pos * n_vel * n_vel
    possible_q1 = np.linspace(conf.lowerPositionLimit_q1, conf.upperPositionLimit_q1, num=n_pos)
    possible_v1 = np.linspace(conf.lowerVelocityLimit_v1, conf.upperVelocityLimit_v1, num=n_vel)
    possible_q2 = np.linspace(conf.lowerPositionLimit_q2, conf.upperPositionLimit_q2, num=n_pos)
    possible_v2 = np.linspace(conf.lowerVelocityLimit_v2, conf.upperVelocityLimit_v2, num=n_vel)
    
    state_array = np.zeros((n_pos * n_pos * n_vel * n_vel, 4))
    #state_array = np.zeros((n_ics, 4))

    i = 0
    for q1 in possible_q1:
        for v1 in possible_v1:
            for q2 in possible_q2:
                for v2 in possible_v2:
                    state_array[i, :] = np.array([q1, v1, q2, v2])
                    i += 1

    # Instance of OCP solver
    ocp = OcpDoublePendulum()

    # Function definition to run in a process
    def ocp_function_double_pendulum(index):
        # I create empy lists to store viable and non viable states
        viable = []
        no_viable = []
        start_index = int(index[0])
        end_index = int(index[1])
    
        # We divide the states grid in complementary subsets
        for i in range(start_index, end_index):
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

    if (multiproc == 1):
        # I subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, n_ics, num=num_processes+1)

        # I define the arguments to pass to the functions: the indexes necessary to split the states grid
        args = []
        i = 0
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

        # Save the results .mat
        mat_file_path_viable = 'data_double.mat'
        data_dict_viable = {'viable_states': viable_states, 'non_viable_states': no_viable_states}
        scipy.io.savemat(mat_file_path_viable, data_dict_viable)

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
        state = state_array[0,:]
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

    if(plot1 == 1):

        # Concatenate viable_states and no_viable_states
        all_states = np.vstack((viable_states, no_viable_states))

        # Create a color array to distinguish between viable and no_viable states
        colors = ['red'] * len(viable_states) + ['blue'] * len(no_viable_states)

        # Function to handle click on the first plot
        def on_first_plot_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                second_plot(all_states, x, y, colors)

        # Function for the second plot
        def second_plot(all_states, x, y, colors, tolerance=0.1):
            fig, ax = plt.subplots()
            ax.set_title(f'Second Plot - Selected Point: ({x}, {y})')

            # Calculate Euclidean distance between the clicked point and all points in q1_v1
            distances = np.linalg.norm(all_states[:, :2] - np.array([x, y]), axis=1)

            # Filter points based on tolerance
            selected_point_indices = np.where(distances < tolerance)[0]
            q2_v2 = all_states[selected_point_indices, 2:]

            # Generate all possible combinations of q2 and v2
            q2_values, v2_values = np.unique(q2_v2[:, 0]), np.unique(q2_v2[:, 1])

            # Create all possible combinations of q2 and v2
            combinations = np.array(np.meshgrid(q2_values, v2_values)).T.reshape(-1, 2)

            # Plot points in the second plot with colors based on viability
            for i, combination in enumerate(combinations):
                color = 'red' if selected_point_indices[i] < len(viable_states) else 'blue'
                ax.scatter(combination[0], combination[1], color=color)
                ax.legend()
                plt.show()

    # First plot
    fig, ax = plt.subplots()
    ax.set_title('First Plot')

    # Plot points for q1 and v1 with colors based on viability
    ax.scatter(all_states[:, 0], all_states[:, 1], c=colors, label='(q1, v1)')

    ax.legend()
    fig.canvas.mpl_connect('button_press_event', on_first_plot_click)
    plt.show()
