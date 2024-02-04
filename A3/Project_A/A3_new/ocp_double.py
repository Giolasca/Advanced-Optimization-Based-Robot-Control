import numpy as np 
import casadi
import doublependulum_dynamics 
import pandas as pd

class OcpSinglePendulum:

    def __init__(self):
        self.dt = dt
        self.w_v1 = w_v1
        self.w_u1 = w_u1
        self.w_v2 = w_v2
        self.w_u2 = w_u2
        self.q_min1 = q_min1
        self.q_max1 = q_max1
        self.v_min1 = v_min1
        self.v_max1 = v_max1
        self.u_min1 = u_min1
        self.u_max1 = u_max1
        self.q_min2 = q_min2
        self.q_max2 = q_max2
        self.v_min2 = v_min2
        self.v_max2 = v_max2
        self.u_min2 = u_min2
        self.u_max2 = u_max2
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
        self.q1 = self.opti.variable(N+1)
        self.v1 = self.opti.variable(N+1)
        self.u1 = self.opti.variable(N)
        self.q2 = self.opti.variable(N+1)
        self.v2 = self.opti.variable(N+1)
        self.u2 = self.opti.variable(N)
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1
        q2 = self.q2
        v2 = self.v2
        u2 = self.u2

        # State Vector initialization
        if X_guess is not None:
            self.opti.set_initial(self.q1, X_guess[:, 0])
            self.opti.set_initial(self.v1, X_guess[:, 1])
            self.opti.set_initial(self.q2, X_guess[:, 0])
            self.opti.set_initial(self.v2, X_guess[:, 1])
        else:
            self.opti.set_initial(self.q1, x_init[0])
            self.opti.set_initial(self.v1, x_init[1])
            self.opti.set_initial(self.q2, x_init[2])
            self.opti.set_initial(self.v2, x_init[3])

        # Control input initialization
        if U_guess is not None:
            for i in range(N):
                self.opti.set_initial(u1[i], U_guess[0, i])
                self.opti.set_initial(u2[i], U_guess[1, i])

        i = 0
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u1 * u1[i]**2 + self.w_u2 * u2[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics Constraint
        for i in range(N):
            x_next = doublependulum_dynamics.f(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array(u1[i], u2[i]))
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(v1[i+1] == x_next[1])
            self.opti.subject_to(q2[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])


        # Initial state constraint
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])

        # State bound constraints
        for i in range(N+1):
            self.opti.subject_to(self.opti.bounded(self.q_min1, q1[i], self.q_max1))
            self.opti.subject_to(self.opti.bounded(self.v_min1, v1[i], self.v_max1))
            self.opti.subject_to(self.opti.bounded(self.q_min2, q2[i], self.q_max2))
            self.opti.subject_to(self.opti.bounded(self.v_min2, v2[i], self.v_max2))
        
        #  Control bounds Constraints
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(self.u_min1, u1[i], self.u_max1))
            self.opti.subject_to(self.opti.bounded(self.u_min2, u2[i], self.u_max2))

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()

        # Store initial state and optimal cost in buffers
        self.buffer_x0.append([x_init[0], x_init[1], x_init[2], x_init[3]])
        self.buffer_cost.append(sol.value(self.cost))

        return sol

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 1.0          # OCP horizon size
    dt = 0.01        # time step
    N = int(T/dt);                  # horizon size

    q_min1, q_max1 = 3/4*np.pi, 5/4*np.pi  # position bounds
    v_min1, v_max1 = -10, 10  # velocity bounds
    u_min1 = -9.81      # min control input
    u_max1= 9.81       # max control input

    q_min2, q_max2 = 3/4*np.pi, 5/4*np.pi  # position bounds
    v_min2, v_max2 = -10, 10  # velocity bounds
    u_min2 = -9.81      # min control input
    u_max2= 9.81       # max control input

    w_u1 = 1e-4
    w_v1 = 1e-2
    w_u2 = 1e-4
    w_v2 = 1e-2
    
    ocp = OcpSinglePendulum()

    n_pos_q1 = 21
    n_vel_v1 = 21
    n_pos_q2 = 21
    n_vel_v2 = 21

    n_ics = n_pos_q1 * n_pos_q2 * n_vel_v1 * n_vel_v2
    possible_q1 = np.linspace(q_min1, q_max1, num=n_pos_q1)
    possible_v1 = np.linspace(v_min1, v_max1, num=n_vel_v1)
    possible_q2 = np.linspace(q_min2, q_max2, num=n_pos_q2)
    possible_v2 = np.linspace(v_min2, v_max2, num=n_vel_v2)
    state_array = np.zeros((n_ics, 4))
    
    i = 0
    for q1 in possible_q1:
        for v1 in possible_v1:
            for q2 in possible_q2:
                for v2 in possible_v2:
                    state_array[i, :] = np.array([q1, v1, q2, v2])
                    i += 1


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