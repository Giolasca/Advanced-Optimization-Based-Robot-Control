import numpy as np
import casadi
import DP_dynamics as DP_dynamics
import ocp_DP_conf as config
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
import time


class SinglePendulumOCP:

    def __init__(self):
        self.N = config.N                       # OCP horizon
        self.w_q1 = config.w_q1                 # weight on first link position
        self.w_u1 = config.w_u1                 # weight on first link control
        self.w_v1 = config.w_v1                 # weight on first link velocity  

        self.w_q2 = config.w_q2                 # weight on second link position
        self.w_u2 = config.w_u2                 # weight on second link control
        self.w_v2 = config.w_v2                 # weight on second link velocity  

    def save_results(self, state_buffer, cost_buffer):        # Save results in a csv file to create the DataSet
        filename = 'ocp_data_DP_target_180_180_run_4_unconstr.csv'
        positions_q1 = [state[0] for state in state_buffer]
        velocities_v1 = [state[1] for state in state_buffer]
        positions_q2 = [state[2] for state in state_buffer]
        velocities_v2 = [state[3] for state in state_buffer]
        df = pd.DataFrame({'q1': positions_q1, 'v1': velocities_v1, 'q2': positions_q2, 'v2': velocities_v2, 'cost': cost_buffer})
        df.to_csv(filename, index=False)

    def solve_ocp(self, initial_state, state_guess=None, control_guess=None):
        self.opti = casadi.Opti()
        
        # Declaration of the variables in casaDi types
        q1 = self.opti.variable(self.N + 1)
        v1 = self.opti.variable(self.N + 1)
        u1 = self.opti.variable(self.N)

        q2 = self.opti.variable(self.N + 1)
        v2 = self.opti.variable(self.N + 1)
        u2 = self.opti.variable(self.N)
        
        # Target position
        q1_des = config.q1_target
        q2_des = config.q2_target

        # State Vector initialization
        if state_guess is not None:
            for i in range(self.N + 1):
                self.opti.set_initial(q1[i], state_guess[0, i])
                self.opti.set_initial(v1[i], state_guess[1, i])
                self.opti.set_initial(q2[i], state_guess[2, i])
                self.opti.set_initial(v2[i], state_guess[3, i])
        else:
            for i in range(self.N + 1):
                self.opti.set_initial(q1[i], initial_state[0])
                self.opti.set_initial(v1[i], initial_state[1])
                self.opti.set_initial(q2[i], initial_state[2])
                self.opti.set_initial(v2[i], initial_state[3])

        # Control input initialization
        if control_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(u1[i], control_guess[0, i])
                self.opti.set_initial(u2[i], control_guess[1, i])

        # Cost functional
        self.total_cost = 0
        self.running_cost = [None,]*(self.N+1)
        for i in range(self.N + 1):
            self.running_cost[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            self.running_cost[i] += self.w_q1 * (q1[i] - q1_des)**2 + self.w_q2 * (q2[i] - q2_des)**2
            if (i<self.N):
                self.running_cost[i] += self.w_u1 * u1[i]**2 + self.w_u2 * u2[i]**2
            self.total_cost += self.running_cost[i]
        self.opti.minimize(self.total_cost)

        # Dynamic constraint
        for i in range(self.N):
            next_state = DP_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i],u2[i]]))
            self.opti.subject_to(q1[i + 1] == next_state[0])
            self.opti.subject_to(v1[i + 1] == next_state[1])
            self.opti.subject_to(q2[i + 1] == next_state[2])
            self.opti.subject_to(v2[i + 1] == next_state[3])

        # Initial state constraint
        self.opti.subject_to(q1[0] == initial_state[0])
        self.opti.subject_to(v1[0] == initial_state[1])
        self.opti.subject_to(q2[0] == initial_state[2])
        self.opti.subject_to(v2[0] == initial_state[3])

        '''
        # Boundary constraints for position and velocity
        for i in range(self.N + 1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(config.q_min, position[i], config.q_max))
            
            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(config.v_min, velocity[i], config.v_max))
            
            # Control bounds
            if i < self.N:
                self.opti.subject_to(self.opti.bounded(config.u_min, control[i], config.u_max))
        '''
        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        return self.opti.solve()

def ocp_task(index_range, ocp_solver, initial_states):
    state_buffer = []       # Buffer to store initial states
    cost_buffer = []        # Buffer to store optimal costs
    
    # Divide the states grid in complementary subsets
    for i in range(index_range[0], index_range[1]):
        initial_state = initial_states[i, :]
        try:
            solution = ocp_solver.solve_ocp(initial_state)
            state_buffer.append([initial_state[0], initial_state[1], initial_state[2], initial_state[3]])
            cost_buffer.append(solution.value(ocp_solver.total_cost))
            print(f"Initial State: [{initial_state[0]:.3f}  {initial_state[1]:.3f} {initial_state[2]:.3f}  {initial_state[3]:.3f}] Cost: {cost_buffer[-1]:.3f}")
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print(f"Could not solve for: [{initial_state[0]:.3f}  {initial_state[1]:.3f} {initial_state[2]:.3f}  {initial_state[3]:.3f}]")
            else:
                print("Runtime error:", e)
    return state_buffer, cost_buffer


if __name__ == "__main__":
    # Instance of OCP solver
    ocp_solver = SinglePendulumOCP()

    if config.grid == 1:
        num_q1 = 21
        num_v1 = 21
        num_q2 = 21
        num_v2 = 21
        total_initial_conditions = num_q1 * num_v1 * num_q2 * num_v2
        initial_states = config.grid_states(num_q1, num_v1, num_q2, num_v2)
    else:
        num_random = 100
        initial_states = config.random_states(num_random)

    # Multi process execution
    if config.multiproc == 1:
        print("Multiprocessing execution started, number of processes:", config.num_processes)
        print("Total points: {}  Calculated points: {}".format(config.tot_points, config.end_index))

        # Subdivide the states grid in equal spaces proportional to the number of processes
        indexes = np.linspace(0, initial_states.shape[0], num=config.num_processes + 1)
        args = [[int(indexes[i]), int(indexes[i + 1])] for i in range(config.num_processes)]
        pool = multiprocessing.Pool(processes=config.num_processes)
        
        # Start execution time
        start_time = time.time()    
        
        # Multiprocess start
        results = pool.starmap(ocp_task, [(args, ocp_solver, initial_states) for args in args])
        
        # Multiprocess end
        pool.close()
        pool.join()

        # End execution time
        end_time = time.time()

        # Store the results
        combined_state_buffer = []
        combined_cost_buffer = []
        for result in results:
            combined_state_buffer.extend(result[0])
            combined_cost_buffer.extend(result[1])

    else:
        print("Single process execution")

        # Full range of indices for single process
        index_range = (0, len(initial_states))

        # Start execution time
        start_time = time.time()

        # Process all initial states in a single call to ocp_task
        state_buffer, cost_buffer = ocp_task(index_range, ocp_solver, initial_states)
        
        # End execution time
        end_time = time.time()

        # Store the results
        combined_state_buffer = state_buffer
        combined_cost_buffer = cost_buffer

    # Time in nice format
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

    # Save data in a .csv file
    ocp_solver.save_results(combined_state_buffer, combined_cost_buffer)