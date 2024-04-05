#!/usr/bin/env python3
import numpy as np
import casadi as ca
from builders import *


barriers = {}


# ============ Task 1 =================
symbolic_state = ca.MX.sym('state_1', 2)
initial_state = {1: np.array([0, 0])}
goal = np.array([6, 6])

predicate = go_to_goal_predicate_2d(goal=goal, epsilon=1, position=symbolic_state)
always = AlwaysOperator(time_interval=TimeInterval(a=20, b=55))
task = StlTask(predicate=predicate, temporal_operator=always)

scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

barrier1 = create_barrier_from_task(task=task, initial_conditions=initial_state, alpha_function=alpha_fun)
# ====================================


# ============ Task 2 =================
symbolic_state1 = ca.MX.sym('state_1', 2)
symbolic_state2 = ca.MX.sym('state_2', 2)
initial_states = {1: np.array([0, -2]), 2: np.array([0, 0])}

predicate = formation_predicate(epsilon=0.1, position_i = symbolic_state2, position_j = symbolic_state1, relative_pos=np.array([1,1])) # relative_pos is the added coord of the robot you want to follow
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)

scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

barrier2 = create_barrier_from_task(task=task, initial_conditions=initial_states, alpha_function=alpha_fun)
# ====================================


# ============ Task 3 =================
symbolic_state1 = ca.MX.sym('state_1', 2)
symbolic_state3 = ca.MX.sym('state_3', 2)
initial_states = {1: np.array([0, 2]), 2: np.array([0, 0])}

predicate = formation_predicate(epsilon=0.1, position_i = symbolic_state3, position_j = symbolic_state1, relative_pos=np.array([1,-1])) # relative_pos is the added coord of the robot you want to follow
always    = AlwaysOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)

scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

barrier3 = create_barrier_from_task(task=task, initial_conditions=initial_states, alpha_function=alpha_fun)
# ====================================


barriers[1] = barrier1
barriers[2] = barrier2
barriers[3] = barrier3