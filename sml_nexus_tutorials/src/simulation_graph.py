#!/usr/bin/env python3
import numpy as np
import casadi as ca
import networkx as nx
from builders import *
import matplotlib.pyplot as plt
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges


# Parameters
barriers = {}
initial_states = {}
edge_barriers = []
scale_factor = 3
communication_radius = 3.0

# Initial states of the robots 
state1 = np.array([0,0]) 
state2 = np.array([0,-2])    
state3 = np.array([0,2])     
initial_states = {1:state1,2:state2,3:state3}

# Creating the robots
robot_1 = Robot(id=1, initial_state=state1)
robot_2 = Robot(id=2, initial_state=state2)
robot_3 = Robot(id=3, initial_state=state3)

# Select desired edge for tasks. Add self loops if you need, because they won't be added otherwise
task_edges = [(1,1),(1,2),(1,3)]
task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
comm_graph = create_communication_graph_from_states(initial_states,communication_radius)

# Creating the alpha function that is the same for all the tasks for now
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])


# ============ Task 1 ====================================================================================================
edge_1    = task_graph[1][1]["container"]
predicate = go_to_goal_predicate_2d(goal=np.array([6, 0]), epsilon=0.1, agent=robot_1)
always = AlwaysOperator(time_interval=TimeInterval(a=10, b=25))
task = StlTask(predicate=predicate, temporal_operator=always)
edge_barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1], alpha_function=alpha_fun))
edge_1.add_tasks(task)

predicate = go_to_goal_predicate_2d(goal=np.array([6, 6]), epsilon=0.1, agent=robot_1)
always = AlwaysOperator(time_interval=TimeInterval(a=40, b=55))
task = StlTask(predicate=predicate, temporal_operator=always)
edge_barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1], alpha_function=alpha_fun))
edge_1.add_tasks(task)

conjoined_barrier = conjunction_of_barriers(edge_barriers, associated_alpha_function=alpha_fun)
barriers[1] = conjoined_barrier
# =========================================================================================================================


# ============ Task 2 =====================================================================================================
edge_12    = task_graph[1][2]["container"]
predicate = formation_predicate(epsilon=0.1, agent_i = robot_2, agent_j = robot_1, relative_pos=np.array([1,1]))
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)
barriers[2] = create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_2], alpha_function=alpha_fun)
edge_12.add_tasks(task)
# =========================================================================================================================


# ============ Task 3 =====================================================================================================
edge_13    = task_graph[1][3]["container"]
predicate = formation_predicate(epsilon=0.1, agent_i = robot_3, agent_j = robot_1, relative_pos=np.array([1,-1])) 
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)
barriers[3] = create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_3], alpha_function=alpha_fun)
edge_13.add_tasks(task)
# =========================================================================================================================




# fig,ax = plt.subplots(2,1)
# nx.draw(comm_graph, with_labels=True, font_weight='bold',ax=ax[0])
# ax[0].set_title("Communication Graph")
# nx.draw(task_graph, with_labels=True, font_weight='bold',ax=ax[1])
# ax[1].set_title("Task Graph")

# plt.show()