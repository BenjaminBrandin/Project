import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables
x_sym = ca.MX.sym('x', 4)  
t_sym = ca.MX.sym('t')
A = ca.DM.zeros(2, 4)  
b = ca.DM.zeros(2) 

# Initial states for robots
dt = 0.1
tspan = np.arange(0, 30.1, dt)
x = ca.DM([0, 0, 20, 0])  
Q = ca.DM_eye(4)

tau1 = 9 # The time at which the gamma1 function becomes zero
tau2 = 25 # The time at which the gamma2 function becomes zero
tau3 = 15 # The time at which the gamma3 function becomes zero

# Initialize variables to store system states and control input history
num_robots = 2
x_history = np.zeros((2 * num_robots, len(tspan))) 
u_history = np.zeros((2 * num_robots, len(tspan)))
b_history = np.zeros((len(tspan)))


# Define barrier function
h1 = 16 - ca.sumsqr(x_sym[:2] - ca.vertcat(10, 7))  
h2 = 16 - ca.sumsqr(x_sym[2:] - ca.vertcat(15, 7))
ny = 1
h12 = -(1/ny)*np.log(np.exp(-ny*h1) + np.exp(-ny*h2))

h3 = 4 - ca.sumsqr(x_sym[:2] - x_sym[2:])
h4 = 9 - ca.sumsqr(x_sym[:2] - ca.vertcat(10, 11))

gamma01 = 1.4 * ca.fabs(ca.substitute(h12, x_sym, x))
gamma02 = 1.4 * ca.fabs(ca.substitute(h3, x_sym, x))
gamma03 = 1.4 * ca.fabs(ca.substitute(h4, x_sym, x))

gamma1 = ca.if_else(t_sym < tau1, -(gamma01 / tau1) * t_sym + gamma01, 0)
gamma2 = ca.if_else(t_sym < tau2, -(gamma02 / tau2) * t_sym + gamma02, 0)
gamma3 = ca.if_else(t_sym < tau3, -(gamma03 / tau3) * t_sym + gamma03, 0)

barrier_func1 = gamma1 + h12
barrier_func2 = gamma2 + h3
barrier_func3 = gamma3 + h4

ny = 1
barrier_func = -(1/ny)*np.log(np.exp(-ny*barrier_func1) + np.exp(-ny*barrier_func2) + np.exp(-ny*barrier_func3))


# Compute the Jacobian of the barrier function with respect to x and t
barrier_jacobian_x = ca.jacobian(barrier_func, x_sym)
barrier_jacobian_t = ca.jacobian(barrier_func, t_sym)

# Create CasADi function for barrier function and its Jacobians
barrier_fn = ca.Function('barrier_fn', [x_sym, t_sym], [barrier_func])
barrier_jacx_fn = ca.Function('barrier_jacx_fn', [x_sym, t_sym], [barrier_jacobian_x])
barrier_jact_fn = ca.Function('barrier_jact_fn', [x_sym, t_sym], [barrier_jacobian_t])

# Simulate system with quadratic programming-based control law
for i, t in enumerate(tspan):

    # Compute barrier function value
    barrier_value = barrier_fn(x, t)
    A = barrier_jacx_fn(x, t)
    b = barrier_jact_fn(x, t) + 1 * barrier_value

    if np.linalg.norm(A) < 1e-10:
        u_hat = ca.DM([0, 0, 0, 0])
    else:
        # Defining the problem and the inequality constraints
        u_hat = ca.MX.sym('u_hat', 4)
        objective = u_hat.T @ Q @ u_hat
        constraint = A @ u_hat + b 
        qp = {'x':u_hat, 'f':objective, 'g':constraint}

        # Solving the quadratic program
        qpsolver = ca.qpsol('u_opt', 'qpoases', qp) 
        u_opt = qpsolver(lbg=0)

        # Extract control input
        u_hat = u_opt['x']

    # # Update system state
    x = x + u_hat*dt

    # # Store current state in history
    x_history[:, i] = x.full().flatten()
    u_history[:, i] = u_hat.full().flatten()
    b_history[i] = barrier_value

# Plot the system trajectory
plt.figure(1)
plt.plot(x_history[0, :], x_history[1, :], x_history[2, :], x_history[3, :])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Robot 1', 'Robot 2'])
plt.title('System Trajectory')
plt.axis('equal')

# Plot circles representing the regions the robots need to be inside
circle1 = plt.Circle((10, 7), 4, color='r', fill=False, linestyle='--')
circle2 = plt.Circle((15, 7), 4, color='b', fill=False, linestyle='--')
circle3 = plt.Circle((10, 11), 3, color='g', fill=False, linestyle='--')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)

# Plot the control barrier function
plt.figure(3)
plt.plot(tspan, b_history)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Control Barrier Function')

# # Plot the control input over time
# plt.figure(2)
# plt.plot(tspan, u_history[0, :], tspan, u_history[1, :], tspan, u_history[2, :], tspan, u_history[3, :])
# plt.legend(['u1', 'u2', 'u3', 'u4'])
# plt.xlabel('Time')
# plt.ylabel('Control Input')
# plt.title('Control Input Over Time')

plt.show()
