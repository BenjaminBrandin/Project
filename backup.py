import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables
x_sym = ca.MX.sym('x', 2)  
t_sym = ca.MX.sym('t')
A = ca.DM.zeros(1, 2)  
b = ca.DM.zeros(1) 

# Initial states for robots
dt = 0.1
tspan = np.arange(0, 30.1, dt)
x = ca.DM([0, 0])  
Q = ca.DM_eye(2)

tau = 15 # The time at which the gamma function becomes zero

# Initialize variables to store system states and control input history
num_robots = 1
x_history = np.zeros((2 * num_robots, len(tspan))) 
u_history = np.zeros((2 * num_robots, len(tspan)))
b_history = np.zeros((len(tspan)))

# Define barrier function
h = 9 - ca.sumsqr(x_sym[:2] - ca.vertcat(10, 11))
gamma0 = 1.4 * ca.fabs(ca.substitute(h, x_sym, x))
gamma = ca.if_else(t_sym < tau, -(gamma0 / tau) * t_sym + gamma0, 0)
barrier_func = gamma + h


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
        u_hat = ca.DM([0, 0])
    else:
        # Defining the problem and the inequality constraints
        u_hat = ca.MX.sym('u_hat', 2)
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
plt.plot(x_history[0, :], x_history[1, :])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Robot 1'])
plt.title('System Trajectory')
plt.axis('equal')

# Plot circles representing the regions the robots need to be inside

circle = plt.Circle((10, 11), 3, color='g', fill=False, linestyle='--')

plt.gca().add_patch(circle)

# Plot the control barrier function
plt.figure(3)
plt.plot(tspan, b_history)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Control Barrier Function')


plt.show()
