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
  <img src="images/osc_example.png" alt="Operational Space Controller" style="width: 45%;"/>
  <img src="images/ic_example.gif" alt="Impedance Control" style="width: 45%;"/>
</div>

