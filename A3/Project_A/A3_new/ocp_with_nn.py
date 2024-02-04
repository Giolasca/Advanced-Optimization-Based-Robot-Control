import numpy as np 
import casadi
import single_pendulum_dynamics 
import pandas as pd
from nn_single import create_model
from tensorflow.keras.models import load_model

class OcpSinglePendulum:

    def __init__(self):
        self.dt = dt
        self.w_v = w_v
        self.w_u = w_u
        self.q_min = q_min
        self.q_max = q_max
        self.v_min = v_min
        self.v_max = v_max
        self.u_min = u_min
        self.u_max = u_max
        self.model = create_model(2)        # Template of NN
        self.model.load_weights("ocp_nn_model.h5")
        self.weights = self.model.get_weights()
        self.buffer_cost = []  # Buffer to store optimal costs
        self.buffer_x0 = []    # Buffer to store initial states
    
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

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = self.w_v * v[i]**2
            if (i < N):
                self.running_costs[i] += self.w_u * u[i]**2
            self.cost += self.running_costs[i]
        
        # Neural network cost term (terminal cost)
        x_final = casadi.vertcat(q[-1], v[-1])
        neural_net_cost = self.neural_network(x_final)
        self.cost += neural_net_cost

        self.opti.minimize(self.cost)

        # Dynamics Constraint
        for i in range(N):
            x_next = single_pendulum_dynamics.f(np.array([q[i], v[i]]), u[i])
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

    T = 1.0          # OCP horizon size
    dt = 0.01        # time step
    N = int(T/dt);                  # horizon size

    q_min, q_max = 3/4*np.pi, 5/4*np.pi  # position bounds
    v_min, v_max = -10, 10  # velocity bounds
    u_min = -9.81      # min control input
    u_max = 9.81       # max control input

    w_u = 1e-4
    w_v = 1e-2
    
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

    for state in state_array:
        try:
            sol = ocp.solve(state, N)
            print("State", state, "Cost", ocp.buffer_cost[-1])
        except Exception as e:
            print(f"Could not solve OCP for {state}: {str(e)}")

    ocp.save_to_csv()

   # Plotting the cost over the initial states
    plt.scatter(np.array(ocp.buffer_x0)[:, 0], np.array(ocp.buffer_x0)[:, 1], c=ocp.buffer_cost, cmap='viridis')
    plt.xlabel('Initial Position (q)')
    plt.ylabel('Initial Velocity (v)')
    plt.title('Cost over Initial States')
    plt.colorbar(label='Cost')
    plt.show()