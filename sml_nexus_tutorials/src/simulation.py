#!/usr/bin/env python3
import numpy as np
import casadi as ca
from builders import *


# Creating the robots
robot_1 = Robot(id=1, initial_state=np.array([0, 0]))
robot_2 = Robot(id=2, initial_state=np.array([0, -2]))
robot_3 = Robot(id=3, initial_state=np.array([0, 2]))

# Creating the alpha function that is the same for all the tasks for now
scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

# Creating the dictionary that will store the barriers
barriers = {}


# ============ Task 1 =====================================================================================================
predicate = go_to_goal_predicate_2d(goal=np.array([6, 1]), epsilon=0.1, agent=robot_1)
always = AlwaysOperator(time_interval=TimeInterval(a=20, b=55))
task = StlTask(predicate=predicate, temporal_operator=always)
barriers[1] = create_barrier_from_task(task=task, initial_conditions=[robot_1], alpha_function=alpha_fun)
# =========================================================================================================================


# ============ Task 2 =====================================================================================================
predicate = formation_predicate(epsilon=0.1, agent_i = robot_2, agent_j = robot_1, relative_pos=np.array([1,1]))
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)
barriers[2] = create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_2], alpha_function=alpha_fun)
# =========================================================================================================================


# ============ Task 3 =====================================================================================================
predicate = formation_predicate(epsilon=0.1, agent_i = robot_3, agent_j = robot_1, relative_pos=np.array([1,-1])) 
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)
barriers[3] = create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_3], alpha_function=alpha_fun)
# =========================================================================================================================