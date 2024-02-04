import numpy as np 
import casadi
import SP_dynamics
import pandas as pd
from nn_single import create_model
import mpc_SP_conf as conf

class OcpSinglePendulum:

    def __init__(self):
        self.dt = dt
        self.w_q = w_q
        self.w_v = w_v
        self.w_u = w_u
        self.q_min = q_min
        self.q_max = q_max
        self.v_min = v_min
        self.v_max = v_max
        self.u_min = u_min
        self.u_max = u_max
        self.model = create_model(2)
        self.model.load_weights("ocp_nn_model.h5")
        self.weights = self.model.get_weights()
        self.mean, self.std = conf.init_scaler()

        self.buffer_cost = []  # Buffer to store optimal costs
        self.buffer_x0 = []    # Buffer to store initial states

    def nn_to_casadi(self, params, x):
        out = np.array(x)
        it = 0

        for param in params:
            param = np.array(param.tolist())

            if it % 2 == 0:
                out = out @ param
            else:
                out = param + out
                for i, item in enumerate(out):
                    out[i] = casadi.fmax(0., casadi.MX(out[i]))

            it += 1

        return casadi.MX(out[0])
    
    def save_to_csv(self, filename='ocp_data.csv'):
        data = {'Initial Position (q)': np.array(self.buffer_x0)[:, 0],
                'Initial Velocity (v)': np.array(self.buffer_x0)[:, 1],
                'Cost': self.buffer_cost}
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()
        self.q = self.opti.variable(N+1)
        self.v = self.opti.variable(N+1)
        self.u = self.opti.variable(N)
        q = self.q
        v = self.v
        u = self.u

        # State Vector initialization
        if X_guess is not None:
            self.opti.set_initial(self.q, X_guess[:, 0])
            self.opti.set_initial(self.v, X_guess[:, 1])
        else:
            self.opti.set_initial(self.q, x_init[0])
            self.opti.set_initial(self.v, x_init[1])

        # Control input initialization
        if U_guess is not None:
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i, :])

        # State Normalization 
        state = [(self.q[N] - self.mean[0])/self.std[0], (self.v[N] - self.mean[1])/self.std[1]]

        q_target =  5/4 * np.pi

        # Cost definition
        self.cost = 0
        self.terminal_cost = self.nn_to_casadi(self.weights, state)
        self.running_costs = [None,]*(N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            self.running_costs[i] += self.w_q * (q[i] - q_target)**2
            if (i<N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        self.cost += self.cost + self.terminal_cost
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
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(self.u_min, u[i], self.u_max))

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()

        # Store initial state and optimal cost in buffers
        self.buffer_x0.append([x_init[0], x_init[1]])
        self.buffer_cost.append(sol.value(self.cost))

        return sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 0.05          # OCP horizon size
    dt = 0.01        # time step
    N = int(T/dt);                  # horizon size

    mpc_step = 200

    q_min, q_max = 3/4*np.pi, 5/4*np.pi  # position bounds
    v_min, v_max = -10, 10  # velocity bounds
    u_min = -9.81      # min control input
    u_max = 9.81       # max control input

    w_q = 1e2
    w_u = 1e-4
    w_v = 1e-1
    
    ocp = OcpSinglePendulum()

    n_pos = 121
    n_vel = 121
    n_ics = n_pos*n_vel
    possible_q = np.linspace(q_min, q_max, num=n_pos)
    possible_v = np.linspace(v_min, v_max, num=n_vel)
    state_array = np.zeros((n_ics, 2))

    j = k = 0
    for i in range(n_ics):
        state_array[i,:] = np.array([possible_q[j], possible_v[k]])
        # x_0_arr[i,:] = np.array([possible_q[i], 0])
        k += 1
        if (k == n_vel):
            k = 0
            j += 1

    initial_state = np.array([3/4*np.pi, 0])
    traj = []
    u = []

    X_guess_new = np.zeros((2, N+1))
    U_guess_new = np.zeros((N))

    sol = ocp.solve(initial_state, N)
    traj.append(np.array([sol.value(ocp.q[1]), sol.value(ocp.v[1])]))
    u.append(sol.value(ocp.u[0]))


    # Create new X_guess
    for i in range(N):
        X_guess_new[0,i] = sol.value(ocp.q[i+1])
        X_guess_new[1,i] = sol.value(ocp.v[i+1])

    # Create new U_guess
    for i in range(N-1):
        U_guess_new[i] = sol.value(ocp.u[i+1])

    for i in range(mpc_step):
        try:
            sol = ocp.solve(initial_state, X_guess_new, U_guess_new, N)
        except Exception as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("")
                print("======================================")
                print("MPC stopped due to infeasible problem")
                print("======================================")
                print("")
                print(ocp.opti.debug.show_infeasibilities())
        
        traj.append(np.array([sol.value(ocp.q[1]), sol.value(ocp.v[1])]))
        u.append(sol.value(ocp.u[0]))

        for k in range(N):
            X_guess_new[0, k] = sol.value(ocp.q[k+1])
            X_guess_new[1, k] = sol.value(ocp.v[k+1])
        
        for k in range(N-1):
            U_guess_new[k] = sol.value(ocp.u[k+1])
        
        print("Step", i+1, "of ", mpc_step, "done")
    
    pos = []
    vel = []

    for i, state in enumerate(traj):
        pos.append(traj[i][0])
        vel.append(traj[i][1])

    for element in traj:
        pos.append(element[0])
        vel.append(element[1])

    # Position plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(pos)
    plt.xlabel('mpc step')
    plt.ylabel('q [rad]')
    plt.title('Position')
    plt.show()

    # Velocity plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(vel)
    plt.xlabel('mpc step')
    plt.ylabel('v [rad/s]')
    plt.title('Velocity')
    plt.show()

    # Torque plot
    fig = plt.figure(figsize=(12,8))
    plt.plot(u)
    plt.xlabel('mpc step')
    plt.ylabel('u [N/m]')
    plt.title('Torque')
    plt.show()