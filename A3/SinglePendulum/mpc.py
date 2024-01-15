import numpy as np
import casadi
import mpc_single_pendulum_conf as conf
import single_pendulum_dynamics
from nn_pytorch import PendulumClassifier
import torch

class MpcSinglePendulum:

    def __init__(self):
        self.T = conf.T                     # MPC horizon
        self.dt = conf.dt                   # time step
        self.w_q = conf.w_q                 # Position weight
        self.w_u = conf.w_u                 # Input weight
        self.w_v = conf.w_v                 # Velocity weight
        self.N = int(self.T/self.dt)        # I initalize the Opti helper from casadi
        self.input_size = conf.input_size
        self.hidden_size1 = conf.hidden_size1
        self.hidden_size2 = conf.hidden_size2
        # Upload the model 
        self.neural_network = PendulumClassifier(self.input_size, self.hidden_size1, self.hidden_size2)
        self.neural_network.load_state_dict(torch.load('pendulum_model.pth'))
        self.neural_network.eval()

    def solve(self, x_init, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()

        # Casadi variables declaration
        q = self.opti.variable(self.N + 1)
        v = self.opti.variable(self.N + 1)
        u = self.opti.variable(self.N)

        # State vector initialization
        if X_guess is not None:
            for i in range(self.N + 1):
                self.opti.set_initial(q[i], X_guess[0, i])
                self.opti.set_initial(v[i], X_guess[1, i])
        else:
            for i in range(self.N + 1):
                self.opti.set_initial(q[i], x_init[0])
                self.opti.set_initial(v[i], x_init[1])

        # Control input vector initialization
        if U_guess is not None:
            for i in range(self.N):
                self.opti.set_initial(u[i], U_guess[i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Target position
        q_target = conf.q_target

        # Cost definition
        cost = 0
        running_costs = [None, ] * (self.N + 1)
        for i in range(self.N + 1):
            running_costs[i] = self.w_v * v[i] ** 2
            running_costs[i] += self.w_q * (q[i] - q_target) ** 2
            if i < self.N:
                running_costs[i] += self.w_u * u[i] ** 2
            cost += running_costs[i]
        self.opti.minimize(cost)

        # Dynamics constraint
        for i in range(self.N):
            x_next = single_pendulum_dynamics.f(np.array([q[i], v[i]]), u[i])
            self.opti.subject_to(q[i + 1] == x_next[0])
            self.opti.subject_to(v[i + 1] == x_next[1])

        # Initial state constraint
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])

        # Bounds constraints
        for i in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit, q[i], conf.upperPositionLimit))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit, v[i], conf.upperVelocityLimit))
            if i < self.N:
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound, u[i], conf.upperControlBound))

        # Terminal constraint
        x_terminal = single_pendulum_dynamics.f(np.array([q[self.N], v[self.N]]), u[self.N-1])
        x_terminal_nn = torch.FloatTensor(conf.scaler.transform(x_terminal))
        terminal_constraint = self.neural_network(x_terminal_nn) >= 0.5
        self.opti.subject_to(terminal_constraint)

        # Solve optimization problem
        return self.opti.solve()



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Instance of OCP solver
    mpc = MpcSinglePendulum()
    
    initial_state = conf.initial_state
    actual_trajectory = []
    actual_inputs = []

    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((2, mpc.N+1))
    new_input_guess = np.zeros((mpc.N))

    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
    actual_inputs.append(sol.value(mpc.u[0]))
    # Dobbiamo creare una state_guess 2 x N+1
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q[i+1])
        new_state_guess[1, i] = sol.value(mpc.v[i+1])
    # Dobbiamo creare una input_guess di dimensione N
    for i in range(mpc.N-1):
        new_input_guess[i] = sol.value(mpc.u[i+1])
        
    for i in range(mpc_step):
        noise = np.random.normal(conf.mean, conf.std, actual_trajectory[i].shape)
        if conf.noise:
            init_state = actual_trajectory[i] + noise
        else:
            init_state = actual_trajectory[i]
        try:
            sol = mpc.solve(init_state, new_state_guess, new_input_guess)
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("")
                print("======================================")
                print("MPC stopped due to infeasible problem")
                print("======================================")
                print("")
                print(mpc.opti.debug.show_infeasibilities())
                break
            else:
                print(e)
        actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
        actual_inputs.append(sol.value(mpc.u[0]))
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q[k+1])
            new_state_guess[1, k] = sol.value(mpc.v[k+1])
        for j in range(mpc.N-1):
            new_input_guess[j] = sol.value(mpc.u[j+1])
        print("Step", i+1, "out of", mpc_step, "done")
        
    positions = []
    velocities = []

    for i, state in enumerate(actual_trajectory):
        positions.append(actual_trajectory[i][0])
        velocities.append(actual_trajectory[i][1])

    # Utilizza il modello PyTorch per ottenere le predizioni
    nn_inputs = np.array(actual_trajectory)
    nn_inputs = conf.scaler.transform(nn_inputs)
    nn_inputs = torch.FloatTensor(nn_inputs)

    with torch.no_grad():
        nn_outputs = mpc.neural_network(nn_inputs)
        predictions = torch.sigmoid(nn_outputs)
        predicted_labels = (predictions >= 0.5).float()

    viable_states = np.array([actual_trajectory[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1])
    no_viable_states = np.array([actual_trajectory[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 0])

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.scatter(positions, velocities, c='g')
    ax.scatter(viable_states[:,0], viable_states[:,1], c='r')
    ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b')
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

    # Torque plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(actual_inputs)
    plt.show()

    positions = []
    velocities = []

    for element in actual_trajectory:
        positions.append(element[0])
        velocities.append(element[1])

    # Position plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(positions)
    plt.show()

    # Velocity plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(velocities)
    plt.show()

