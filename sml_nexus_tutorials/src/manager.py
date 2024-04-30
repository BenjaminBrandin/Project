#!/usr/bin/env python3
import rospy
import sys
import os
import yaml
import numpy as np
import casadi as ca
import networkx as nx
from std_msgs.msg import String
from custom_msg.msg import task_msg
from builders import Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, formation_predicate, create_barrier_from_task
import matplotlib.pyplot as plt
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges



# Parameters
barriers = []
robots = {}
agents = []
edge_barriers = []
scale_factor = 3
communication_radius = 3.0


# Load the initial states from the yaml file
with open("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/start_pos.yaml") as file:
    start_pos = yaml.safe_load(file)   # This is used when I want co create class objects: yaml.load(file, Loader=yaml.FullLoader)

# Load the task edges from the yaml file
with open("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/tasks.yaml") as file:
    tasks = yaml.safe_load(file)


# Initial states of the robots and creating the robots
for i, (state_key, state_value) in enumerate(start_pos.items(), start=1):
    state = np.array(state_value)
    agents.append(i)
    robot = Agent(id=i, initial_state=state)
    robots[i] = robot




# Creating the graphs
task_edges = [tuple(task["EDGE"]) for task in tasks.values()]
# Extract the communication information
communication_info = {task_name: {"EDGE": task_info["EDGE"], "COMMUNICATE": task_info["COMMUNICATE"]} for task_name, task_info in tasks.items()}

task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
comm_graph = create_communication_graph_from_states(agents, communication_info)



# Creating the alpha function that is the same for all the tasks for now
dummy_scalar = ca.MX.sym('dummy_scalar', 1)
alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])


# Iterate over the tasks in the yaml file
for task_name, task_info in tasks.items():
    # Get the edge from the task graph
    edge = task_graph[task_info["EDGE"][0]][task_info["EDGE"][1]]["container"]

    # Create the predicate based on the type of the task
    if task_info["TYPE"] == "go_to_goal_predicate_2d":
        predicate = go_to_goal_predicate_2d(goal=np.array(task_info["CENTER"]), epsilon=task_info["EPSILON"], agent=robots[task_info["INVOLVED_AGENTS"][0]])
    elif task_info["TYPE"] == "formation_predicate":
        predicate = formation_predicate(epsilon=task_info["EPSILON"], agent_i=robots[task_info["INVOLVED_AGENTS"][0]], agent_j=robots[task_info["INVOLVED_AGENTS"][1]], relative_pos=np.array(task_info["CENTER"]))
    # Add more predicates here

    # Create the temporal operator
    temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))

    # Create the task
    task = StlTask(predicate=predicate, temporal_operator=temporal_operator)

    # Add the task to the barriers and the edge
    barriers.append(create_barrier_from_task(task=task, initial_conditions=[robots[i] for i in task_info["INVOLVED_AGENTS"]], alpha_function=alpha_fun))
    edge.add_tasks(task)


# def draw_graph(ax, graph, title):
#     """
#     This function draws a graph on a given Axes object and sets its title.

#     Parameters:
#     ax (matplotlib.axes.Axes): The Axes object to draw the graph on.
#     graph (networkx.classes.graph.Graph): The graph to draw.
#     title (str): The title of the graph.
#     """
#     nx.draw(graph, with_labels=True, font_weight='bold', ax=ax)
#     ax.set_title(title)

# # Create the subplots
# fig, ax = plt.subplots(2, 1)
# # Draw the communication graph
# draw_graph(ax[0], comm_graph, "Communication Graph")
# # Draw the task graph
# draw_graph(ax[1], task_graph, "Task Graph")
# # Display the plots
# plt.show()


