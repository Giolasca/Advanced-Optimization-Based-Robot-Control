import numpy as np
import casadi
import SP_dynamics as dynamics
import ocp_SP_conf as config
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
import time


class SinglePendulumOCP:

    def __init__(self):
        self.horizon = config.T                 # Time horizon
        self.time_step = config.dt              # Time step
        self.weight_position = config.w_q       # weight on position
        self.weight_input = config.w_u          # weight on control
        self.weight_velocity = config.w_v       # weight on velocity    

    def save_results(self, state_buffer, cost_buffer, filename):        # Save results in a csv file to create the DataSet
        positions = [state[0] for state in state_buffer]
        velocities = [state[1] for state in state_buffer]
        df = pd.DataFrame({'position': positions, 'velocity': velocities, 'cost': cost_buffer})
        df.to_csv(filename, index=False)

    def solve_ocp(self, initial_state, state_guess=None, control_guess=None):
        self.N = int(self.horizon / self.time_step)
        self.opti = casadi.Opti()
        
        # Declaration of the variables in casaDi types
        position = self.opti.variable(self.N + 1)
        velocity = self.opti.variable(self.N + 1)
        control = self.opti.variable(self.N)
        
        if state_guess is not None:
            for i in range(self.N + 1):
                self.opti.set_initial(position[i], state_guess[0, i])
                self.opti.set_initial(velocity[i], state_guess[1, i])
        else:
            for i in range(self.N + 1):
                self.opti.set_initial(position[i], initial_state[0])
                self.opti.set_initial(velocity[i], initial_state[1])

        if control_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(control[i], control_guess[i])

        # Cost functional
        self.total_cost = 0
        self.running_cost = [None,]*(self.N+1)
        for i in range(self.N + 1):
            self.running_cost[i] = self.weight_velocity * velocity[i] ** 2
            if i < self.N:
                self.running_cost[i] += self.weight_input * control[i] ** 2
            self.total_cost += self.running_cost[i]
        self.opti.minimize(self.total_cost)

        for i in range(self.N):
            # Dynamic constraint
            next_state = dynamics.f(np.array([position[i], velocity[i]]), control[i])
            self.opti.subject_to(position[i + 1] == next_state[0])
            self.opti.subject_to(velocity[i + 1] == next_state[1])

        # Initial state constraint
        self.opti.subject_to(position[0] == initial_state[0])
        self.opti.subject_to(velocity[0] == initial_state[1])

        for i in range(self.N + 1):
            # Boundary constraints for position and velocity
            self.opti.subject_to(self.opti.bounded(config.q_min, position[i], config.q_max))
            self.opti.subject_to(self.opti.bounded(config.v_min, velocity[i], config.v_max))
            if i < self.N:
                self.opti.subject_to(self.opti.bounded(config.u_min, control[i], config.u_max))

        # Choosing solver
        solver_options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        solver_settings = {"max_iter": int(config.max_iter)}
        self.opti.solver("ipopt", solver_options, solver_settings)

        return self.opti.solve()

def ocp_task(index_range, ocp_solver, initial_states):
    state_buffer = []
    cost_buffer = []
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
    
    ocp_solver = SinglePendulumOCP()

    if config.grid == 1:
        num_positions = 121
        num_velocities = 121
        total_initial_conditions = num_positions * num_velocities
        initial_states = config.grid_states(num_positions, num_velocities)
    else:
        num_random = 100
        initial_states = config.random_states(num_random)

    # Multiprocess run
    if config.multiproc == 1:
        print("Multiprocessing execution started, number of processes:", config.num_processes)
        index_ranges = np.linspace(0, total_initial_conditions, num=config.num_processes + 1)
        process_args = [(int(index_ranges[i]), int(index_ranges[i + 1])) for i in range(config.num_processes)]
        pool = multiprocessing.Pool(processes=config.num_processes)
        start_time = time.time()
        results = pool.starmap(ocp_task, [(args, ocp_solver, initial_states) for args in process_args])
        pool.close()
        pool.join()
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")
        combined_state_buffer = []
        combined_cost_buffer = []
        for result in results:
            combined_state_buffer.extend(result[0])
            combined_cost_buffer.extend(result[1])
    else:
        print("Single process execution")
        state_buffer = []
        cost_buffer = []
        start_time = time.time()
        for initial_state in initial_states:
            try:
                solution = ocp_solver.solve_ocp(initial_state)
                state_buffer.append([initial_state[0], initial_state[1]])
                cost_buffer.append(solution.value(ocp_solver.total_cost))
                print("Feasible initial state found:", initial_state, "Cost:", solution.value(ocp_solver.total_cost))
            except RuntimeError as e:
                if "Infeasible_Problem_Detected" in str(e):
                    print("Could not solve for:", initial_state)
                else:
                    print("Runtime error:", e)
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

    # Plot the cost map
    ocp_solver.save_results(combined_state_buffer, combined_cost_buffer, 'ocp_results.csv')
    plt.scatter(np.array(combined_state_buffer)[:, 0], np.array(combined_state_buffer)[:, 1], c=combined_cost_buffer, cmap='viridis')
    plt.xlabel('Initial Position (q)')
    plt.ylabel('Initial Velocity (v)')
    plt.title('Cost over Initial States')
    plt.colorbar(label='Cost')
    plt.show()

