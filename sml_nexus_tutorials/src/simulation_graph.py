#!/usr/bin/env python3
import rospy
import sys
import json
import numpy as np
import casadi as ca
import networkx as nx
from std_msgs.msg import String
from builders import Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, formation_predicate, create_barrier_from_task
import matplotlib.pyplot as plt
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges


# Parameters
barriers = []
initial_states = {}
edge_barriers = []
scale_factor = 3
communication_radius = 3.0

# Initial states of the robots 
state1 = np.array([-3,-3]) 
state2 = np.array([0,-2])    
state3 = np.array([-3,2])     
initial_states = {1:state1,2:state2,3:state3}

# Creating the robots
robot_1 = Agent(id=1, initial_state=state1)
robot_2 = Agent(id=2, initial_state=state2)
robot_3 = Agent(id=3, initial_state=state3)

# Creating the graphs
task_edges = [(1,1),(1,2),(1,3)]
task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
comm_graph = create_communication_graph_from_states(initial_states,communication_radius)

# Creating the alpha function that is the same for all the tasks for now
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

# Add communication maintenance tasks to the task graph

# ============ Task 1 ====================================================================================================
edge_1 = task_graph[1][1]["container"]
predicate = go_to_goal_predicate_2d(goal=np.array([9, 7]), epsilon=2, agent=robot_1)
temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=30, b=50))
task = StlTask(predicate=predicate, temporal_operator=temporal_operator)
barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1], alpha_function=alpha_fun)) 
edge_1.add_tasks(task)
# =========================================================================================================================


# ============ Task 2 =====================================================================================================
edge_12 = task_graph[1][2]["container"]
predicate = formation_predicate(epsilon=1, agent_i = robot_1, agent_j = robot_2, relative_pos=np.array([1.5,-1.5]))
temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=30,b=40)) 
task = StlTask(predicate=predicate,temporal_operator=temporal_operator)
barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_2], alpha_function=alpha_fun)) 
edge_12.add_tasks(task)
# =========================================================================================================================


# ============ Task 3 =====================================================================================================
edge_13 = task_graph[1][3]["container"]
predicate = formation_predicate(epsilon=1, agent_i = robot_1, agent_j = robot_3, relative_pos=np.array([-1.5,1.5])) 
temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=30,b=40)) 
task = StlTask(predicate=predicate,temporal_operator=temporal_operator)
barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_3], alpha_function=alpha_fun))
edge_13.add_tasks(task)
# =========================================================================================================================