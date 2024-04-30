#!/usr/bin/env python3
import rospy
from custom_msg.msg import task_msg
import yaml
import numpy as np
import casadi as ca
import networkx as nx
from builders import Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, formation_predicate, create_barrier_from_task, epsilon_position_closeness_predicate
import matplotlib.pyplot as plt
from graph_module import EdgeTaskContainer, create_communication_graph_from_states, create_task_graph_from_edges
from custom_msg.msg import task_msg
# from decomposition_module import computeNewTaskGraph

class Manager():

    def __init__(self):

        rospy.init_node('manager')

        # Parameters
        self.barriers = []
        self.agents = {}
        self.scale_factor = 3


        # setup publishers
        self.task_pub = rospy.Publisher("tasks", task_msg, queue_size=10)


        # Load the initial states and the task from the yaml files
        with open("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/start_pos.yaml") as file:
            self.start_pos = yaml.safe_load(file)   # Replace this with $(arg coord) in launch file and get_param(-coord) in the code

        with open("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/tasks.yaml") as file:
            self.tasks = yaml.safe_load(file)

        # Initial states of the robots and creating the robots
        for i, (state_key, state_value) in enumerate(self.start_pos.items(), start=1):
            state = np.array(state_value)
            agent = Agent(id=i, initial_state=state)
            self.agents[i] = agent

        # Extracting the edges of the tasks and the communication information
        task_edges = [tuple(task["EDGE"]) for task in self.tasks.values()]
        communication_info = {task_name: {"EDGE": task_info["EDGE"], "COMMUNICATE": task_info["COMMUNICATE"]} for task_name, task_info in self.tasks.items()}

        # Creating the graphs
        self.task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
        self.comm_graph = create_communication_graph_from_states(self.agents.keys(), communication_info)
        
        self.update_graph()

        # self.decomposed_graph = computeNewTaskGraph(self.task_graph, self.comm_graph, task_edges, communication_info)

        self.plot_graph() # If you comment out this line, you need rospy.sleep(6) before self.create_tasks() to ensure the control nodes have loaded
        self.publish_tasks()





    def update_graph(self):
        for task_name, task_info in self.tasks.items():
            # Get the edge from the task graph
            edge: EdgeTaskContainer = self.task_graph[task_info["EDGE"][0]][task_info["EDGE"][1]]["container"]
            # Create the task
            task = self.create_task(task_info)
            # Add the task to the edge
            edge.add_tasks(task)


    def create_task(self, task_info) -> StlTask:
        # Create the predicate based on the type of the task
        if task_info["TYPE"] == "go_to_goal_predicate_2d":
            predicate = go_to_goal_predicate_2d(goal=np.array(task_info["CENTER"]), epsilon=task_info["EPSILON"], agent=self.agents[task_info["INVOLVED_AGENTS"][0]])
        elif task_info["TYPE"] == "formation_predicate":
            predicate = formation_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]], relative_pos=np.array(task_info["CENTER"]))
        elif task_info["TYPE"] == "epsilon_position_closeness_predicate":
            predicate = epsilon_position_closeness_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]])

        # Create the temporal operator
        if task_info["TEMP_OP"] == "AlwaysOperator":
            temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))
        elif task_info["TEMP_OP"] == "EventuallyOperator":
            temporal_operator = EventuallyOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))

        # Create the task
        task = StlTask(predicate=predicate, temporal_operator=temporal_operator)
        return task


    def plot_graph(self):
        fig, ax = plt.subplots(2, 1)
        self.draw_graph(ax[0], self.comm_graph, "Communication Graph")
        self.draw_graph(ax[1], self.task_graph, "Task Graph")
        plt.show()

    def draw_graph(self, ax, graph, title):
        nx.draw(graph, with_labels=True, font_weight='bold', ax=ax)
        ax.set_title(title)


    def publish_tasks(self):
        for task_name, task_value in self.tasks.items():
            # Create a task_msg from the task
            task_message = task_msg()
            # Fill in the fields of task_message based on the structure of task
            task_message.edge = task_value['EDGE']
            task_message.type = task_value['TYPE']
            task_message.center = task_value['CENTER']
            task_message.epsilon = task_value['EPSILON']
            task_message.temp_op = task_value['TEMP_OP']
            task_message.interval = task_value['INTERVAL']
            task_message.involved_agents = task_value['INVOLVED_AGENTS']
            task_message.communicate = task_value['COMMUNICATE']
            # Then publish the message
            self.task_pub.publish(task_message)
            # Sleep for a while to ensure the message is published and processed
            rospy.sleep(1)



if __name__ == "__main__":
    manager = Manager()

    

        