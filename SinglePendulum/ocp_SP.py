import numpy as np
import casadi
import SP_dynamics as SP_dynamics
import ocp_SP_conf as config
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
import time


class SinglePendulumOCP:

    def __init__(self):
        self.N = config.N                       # OCP horizon
        self.weight_position = config.w_q       # weight on position
        self.weight_input = config.w_u          # weight on control
        self.weight_velocity = config.w_v       # weight on velocity    

    def save_results(self, state_buffer, cost_buffer):        # Save results in a csv file to create the DataSet
        filename = 'ocp_data_SP_target_TT.csv'
        positions = [state[0] for state in state_buffer]
        velocities = [state[1] for state in state_buffer]
        df = pd.DataFrame({'position': positions, 'velocity': velocities, 'cost': cost_buffer})
        df.to_csv(filename, index=False)

    def solve_ocp(self, initial_state, state_guess=None, control_guess=None):
        self.opti = casadi.Opti()
        
        # Declaration of the variables in casaDi types
        position = self.opti.variable(self.N + 1)
        velocity = self.opti.variable(self.N + 1)
        control = self.opti.variable(self.N)
        
        # Target position
        q_target = config.q_target

        # State Vector initialization
        if state_guess is not None:
            for i in range(self.N + 1):
                self.opti.set_initial(position[i], state_guess[0, i])
                self.opti.set_initial(velocity[i], state_guess[1, i])
        else:
            for i in range(self.N + 1):
                self.opti.set_initial(position[i], initial_state[0])
                self.opti.set_initial(velocity[i], initial_state[1])

        # Control input initialization
        if control_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(control[i], control_guess[i])

        # Cost functional
        self.total_cost = 0
        self.running_cost = [None,]*(self.N+1)
        for i in range(self.N + 1):
            self.running_cost[i] = self.weight_velocity * velocity[i]**2
            self.running_cost[i] += self.weight_position * (position[i] - q_target)**2
            if (i<self.N):
                self.running_cost[i] += self.weight_input * control[i]**2
            self.total_cost += self.running_cost[i]
        self.opti.minimize(self.total_cost)

        # Dynamic constraint
        for i in range(self.N):
            next_state = SP_dynamics.f(np.array([position[i], velocity[i]]), control[i])
            self.opti.subject_to(position[i + 1] == next_state[0])
            self.opti.subject_to(velocity[i + 1] == next_state[1])

        # Initial state constraint
        self.opti.subject_to(position[0] == initial_state[0])
        self.opti.subject_to(velocity[0] == initial_state[1])

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
            state_buffer.append([initial_state[0], initial_state[1]])
            cost_buffer.append(solution.value(ocp_solver.total_cost))
            print(f"Initial State: [{initial_state[0]:.3f}  {initial_state[1]:.3f}] Cost: {cost_buffer[-1]:.3f}")
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print(f"Could not solve for: [{initial_state[0]:.3f}  {initial_state[1]:.3f}]")
            else:
                print("Runtime error:", e)
    return state_buffer, cost_buffer


if __name__ == "__main__":
    # Instance of OCP solver
    ocp_solver = SinglePendulumOCP()

    if config.grid == 1:
        num_positions = 121
        num_velocities = 121
        total_initial_conditions = num_positions * num_velocities
        initial_states = config.grid_states(num_positions, num_velocities)
    else:
        num_random = 100
        initial_states = config.random_states(num_random)

    # Multi process execution
    if config.multiproc == 1:
        print("Multiprocessing execution started, number of processes:", config.num_processes)
        
        # Subdivide the states grid in equal spaces proportional to the number of processes
        index_ranges = np.linspace(0, total_initial_conditions, num=config.num_processes + 1)
        process_args = [(int(index_ranges[i]), int(index_ranges[i + 1])) for i in range(config.num_processes)]
        pool = multiprocessing.Pool(processes=config.num_processes)
        
        # Start execution time
        start_time = time.time()    
        
        # Multiprocess start
        results = pool.starmap(ocp_task, [(args, ocp_solver, initial_states) for args in process_args])
        
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
    
    # Plot the cost map
    plt.scatter(np.array(combined_state_buffer)[:, 0], np.array(combined_state_buffer)[:, 1], c=combined_cost_buffer, cmap='viridis')
    plt.xlabel('Initial Position (q)')
    plt.ylabel('Initial Velocity (v)')
    plt.title('Cost over Initial States')
    plt.colorbar(label='Cost')
    plt.show()