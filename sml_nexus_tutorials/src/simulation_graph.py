#!/usr/bin/env python3
import numpy as np
import casadi as ca
import networkx as nx
from builders import *
import matplotlib.pyplot as plt
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges

# create a list of unique agents IDs
# agents_ids           = [1,2,3,4,5,6,7]
agents_ids           = [1,2,3]
communication_radius = 8.5
sensing_radius       = 4.


# create initial conditions 
initial_agents_state = {}
pos1 = np.array([0,0]);     state1 = np.array([pos1[0],pos1[1]]) 
pos2 = np.array([0,-2]);    state2 = np.array([pos2[0],pos2[1]])      
pos3 = np.array([0,2]);     state3 = np.array([pos3[0],pos3[1]])       
# pos4 = np.array([-3,6]);    state4 = np.array([pos4[0],pos4[1]])   
# pos5 = np.array([-7,2]);    state5 = np.array([pos5[0],pos5[1]])   
# pos6 = np.array([-7,-6]);   state6 = np.array([pos6[0],pos6[1]])   
# pos7 = np.array([-3,-6]);   state7 = np.array([pos7[0],pos7[1]])   

# initial_agents_state = {1:state1,2:state2,3:state3,4:state4,5:state5,6:state6,7:state7} # create some random intial conditions
initial_agents_state = {1:state1,2:state2,3:state3}
initial_time         = 0.

comm_graph = create_communication_graph_from_states(initial_agents_state,communication_radius)

# Select desired edge for tasks. Add self loops if you need, because they won't be added otherwise
# task_edges = [(1,2),(1,3),(2,4),(2,5),(3,6),(3,7),(1,1)]
task_edges = [(1,1),(1,2),(1,3)]
task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph




barriers = {}


# ============ Task 1 =================
symbolic_state = ca.MX.sym('state_1', 2)
initial_state1 = {1: np.array([0, 0])}
goal = np.array([6, 1])

edge_1    = task_graph[1][1]["container"]
predicate = go_to_goal_predicate_2d(goal=goal, epsilon=1, position=symbolic_state)
always = AlwaysOperator(time_interval=TimeInterval(a=20, b=55))
task = StlTask(predicate=predicate, temporal_operator=always)
edge_1.add_tasks(task)

scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

barrier1 = create_barrier_from_task(task=task, initial_conditions=initial_state1, alpha_function=alpha_fun)
# ====================================


# ============ Task 2 =================
symbolic_state1 = ca.MX.sym('state_1', 2)
symbolic_state2 = ca.MX.sym('state_2', 2)
initial_states2 = {1: np.array([0, 0]), 2: np.array([0, -2])}

edge_12    = task_graph[1][2]["container"]
predicate = formation_predicate(epsilon=0.1, position_i = symbolic_state2, position_j = symbolic_state1, agents = np.array([1, 2]), relative_pos=np.array([1,1])) # relative_pos is the added coord of the robot you want to follow
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)
edge_12.add_tasks(task)

scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

barrier2 = create_barrier_from_task(task=task, initial_conditions=initial_states2, alpha_function=alpha_fun)
# ====================================


# ============ Task 3 =================
symbolic_state1 = ca.MX.sym('state_1', 2)
symbolic_state3 = ca.MX.sym('state_3', 2)
initial_states3 = {1: np.array([0, 0]), 3: np.array([0, 2])}

edge_13    = task_graph[1][3]["container"]
predicate = formation_predicate(epsilon=0.1, position_i = symbolic_state3, position_j = symbolic_state1, agents = np.array([1, 3]),  relative_pos=np.array([1,-1])) # relative_pos is the added coord of the robot you want to follow
always    = EventuallyOperator(time_interval=TimeInterval(a=20,b=30)) 
task      = StlTask(predicate=predicate,temporal_operator=always)
edge_13.add_tasks(task)

scale_factor = 3
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

barrier3 = create_barrier_from_task(task=task, initial_conditions=initial_states3, alpha_function=alpha_fun)
# ====================================


barriers[1] = barrier1
barriers[2] = barrier2
barriers[3] = barrier3


# fig,ax = plt.subplots(2,1)

# nx.draw(comm_graph, with_labels=True, font_weight='bold',ax=ax[0])
# ax[0].set_title("Communication Graph")
# nx.draw(task_graph, with_labels=True, font_weight='bold',ax=ax[1])
# ax[1].set_title("Task Graph")

# plt.show()