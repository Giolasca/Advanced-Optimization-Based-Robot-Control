# Advanced-Optimization-Based-Robot-Control

## A1 
The goal of the assignment is to implement different types of controllers, in particular:
- Operational Space Controller
- Impedance Controller

For the Operational Space Control (OSC), we are interested in testing the control applied to a postural task, which means we want to stabilize the position of the end-effector at a given point and then to a reference trajectory.

For the Impedance Control (IC),we want to compare different formulations (simplified and complete versions) and apply them to a postural task which will be, again, the stabilization at a given point for the end-effector and after we want to stabilize the end-effector around a given trajectory.

## A2
The goal of the assignment is to determine the viability kernel for a simple non linear system such as the single pendulum.

The viability kernel is the largest set of feasible states, starting from which the system can always remain inside the set, therefore without violating the constraints. The viability kernel is the largest control invariant set, and its computation has been the subject of many research papers.

<div style="display: flex; justify-content: space-around;">
  <img src="\A2\Images\OCPs\5g\4500p-5g.jpg" style="width: 47%; margin-right: 6%;"/>
  <img src="\A2\Images\OCPs\1g\4500p_1g_rand.jpg" style="width: 47%; margin-left: 6%;"/>
</div


## A3
The goal of the assignment focuses on learning a value function \( V \) that can then be used as a terminal cost in a Model Predictive Control (MPC) formulation to ensure that the problem to be solved is recursively feasible.

This section details the steps followed during the project:

1. **Formulation of the Problem using CasADi**: CasADi is a software library used for solving Optimal Control Problems (OCPs). The first step involves formulating the problem in CasADi, where we solve many OCPs starting from different initial states, either chosen randomly or on a grid.

2. **Training the Neural Network**: For every solved OCP, we store the initial state \( x_0 \) and the corresponding optimal cost \( J(x_0) \) in a buffer. These data points are then used to train a neural network to predict the optimal cost \( J \) given the initial state \( x_0 \).

3. **Formulation of the MPC**: Once the neural network has been trained, it is used as a terminal cost inside an OCP with the same formulation but with a shorter horizon (e.g., half the original horizon). This step involves reformulating the MPC problem to include the learned terminal cost function to ensure the recursive feasibility of the control strategy.

The aim of this project is to empirically show that the introduction of the terminal cost compensates for the decrease in the horizon length. By learning an approximate value function and incorporating it into the MPC formulation, we aim to maintain or improve the performance of the control system despite the shorter prediction horizon.


### Results for a single pendulum
<div style="display: flex; justify-content: space-around;">
  <img src="A3_A\SinglePendulum\Plots_&_Animations\2D_graph.png" style="width: 47%; margin-right: 6%;"/>
  <img src="A3_A\SinglePendulum\Plots_&_Animations\SinglePendulum.gif" alt="Impedance Control" style="width: 47%; margin-left: 6%;"/>
</div

### Results for a double pendulum
![Trajectory Task](A3_A\DoublePendulum\Plots_&_Animations\DoublePendulum.gif)
