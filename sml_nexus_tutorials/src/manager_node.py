#!/usr/bin/env python3
import yaml
import rospy
import numpy as np
import networkx as nx
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
from custom_msg.msg import task_msg
from decomposition_module import computeNewTaskGraph
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges
from builders import (Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, 
                      formation_predicate, epsilon_position_closeness_predicate, collision_avoidance_predicate)


class Manager():

    def __init__(self):

        rospy.init_node('manager')

        # Parameters
        self.barriers = []
        self.sub_tasks: list[StlTask]= []
        self.agents = {}
        self.scale_factor = 3
        self.total_tasks = 0

        
        # setup publishers
        self.task_pub = rospy.Publisher("tasks", task_msg, queue_size=10)
        self.numOfTasks_pub = rospy.Publisher("numOfTasks", Int32, queue_size=10)

        # Load the initial states and the task from the yaml files
        with open("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/start_pos.yaml") as file:
            self.start_pos = yaml.safe_load(file)  

        with open("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/tasks.yaml") as file:
            self.tasks = yaml.safe_load(file)

        # Initial states of the robots and creating the robots
        start_positions: dict[int, np.ndarray] = {}
        for i, (state_key, state_value) in enumerate(self.start_pos.items(), start=1):
            self.agents[i] = Agent(id=i, initial_state=np.array(state_value))
            start_positions[i] = np.array(state_value)

        # Extracting the edges of the tasks and the communication information
        task_edges = [tuple(task["EDGE"]) for task in self.tasks.values()]
        communication_info = {task_name: {"EDGE": task_info["EDGE"], "COMMUNICATE": task_info["COMMUNICATE"]} for task_name, task_info in self.tasks.items()}

        # Creating the graphs
        self.comm_graph = create_communication_graph_from_states(self.agents.keys(), communication_info) 
        self.task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
        self.initial_task_graph = self.task_graph.copy()

        # Fill the task graph with the tasks and decompose the edges that are not communicative
        self.update_graph()
        computeNewTaskGraph(self.task_graph, self.comm_graph, communication_info, start_position=start_positions)
        
        # publish the tasks
        self.print_tasks()
        self.plot_graph()
        self.publish_numOfTask()
        self.publish_tasks()


    def update_graph(self):
        for task_info in self.tasks.values():
            # Create the task
            task = self.create_task(task_info)
            # Add the task to the edge
            self.task_graph[task_info["EDGE"][0]][task_info["EDGE"][1]]["container"].add_tasks(task)


    def create_task(self, task_info) -> StlTask:
        # Create the predicate based on the type of the task
        if task_info["TYPE"] == "go_to_goal_predicate_2d":
            predicate = go_to_goal_predicate_2d(goal=np.array(task_info["CENTER"]), epsilon=task_info["EPSILON"], 
                                                agent=self.agents[task_info["INVOLVED_AGENTS"][0]])
        elif task_info["TYPE"] == "formation_predicate":
            predicate = formation_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], 
                                            agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]], relative_pos=np.array(task_info["CENTER"]))
        elif task_info["TYPE"] == "epsilon_position_closeness_predicate":
            predicate = epsilon_position_closeness_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], 
                                                             agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]])
        elif task_info["TYPE"] == "collision_avoidance_predicate":
            predicate = collision_avoidance_predicate(epsilon=task_info["EPSILON"], agent_i=self.agents[task_info["INVOLVED_AGENTS"][0]], 
                                                      agent_j=self.agents[task_info["INVOLVED_AGENTS"][1]])
        else:
            raise Exception("Task type" + str(task_info["TYPE"]) + "is not supported")
        
        # Create the temporal operator
        if task_info["TEMP_OP"] == "AlwaysOperator":
            temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))
        elif task_info["TEMP_OP"] == "EventuallyOperator":
            temporal_operator = EventuallyOperator(time_interval=TimeInterval(a=task_info["INTERVAL"][0], b=task_info["INTERVAL"][1]))

        # Create the task
        task = StlTask(predicate=predicate, temporal_operator=temporal_operator)
        return task


    def plot_graph(self):
        fig, ax = plt.subplots(3, 1)
        self.draw_graph(ax[0], self.comm_graph, "Communication Graph")
        self.draw_graph(ax[1], self.initial_task_graph, "Initial Task Graph")
        self.draw_graph(ax[2], self.task_graph, "Decomposed Task Graph")
        plt.show()

    def draw_graph(self, ax, graph, title):
        nx.draw(graph, with_labels=True, font_weight='bold', ax=ax)
        ax.set_title(title)


    def print_tasks(self):
        for i,j,attr in self.task_graph.edges(data=True):
            tasks = attr["container"].task_list
            for task in tasks:
                self.total_tasks += 1
                print("-----------------------------------")
                print(f"EDGE: {list(task.edgeTuple)}")
                print(f"TYPE: {task.type}")
                print(f"CENTER: {task.center}")
                print(f"EPSILON: {task.epsilon}")
                print(f"TEMP_OP: {task.temporal_type}")
                print(f"INTERVAL: {task.time_interval.aslist}")
                print(f"INVOLVED_AGENTS: {task.contributing_agents}")
                print(f"COMMUNICATE: {True}")
                print("-----------------------------------")
        rospy.sleep(0.5)
        

    def publish_tasks(self):
        tasks: list[StlTask] = []
        for i,j,attr in self.task_graph.edges(data=True):
            tasks = attr["container"].task_list
            for task in tasks:
                task_message = task_msg()
                task_message.edge = list(task.edgeTuple)
                task_message.type = task.type                               
                task_message.center = task.center                           
                task_message.epsilon = task.epsilon                         
                task_message.temp_op = task.temporal_type
                task_message.interval = task.time_interval.aslist
                task_message.involved_agents = task.contributing_agents
                task_message.communicate = True                             

                # Then publish the message
                self.task_pub.publish(task_message)
                rospy.sleep(0.5)

    def publish_numOfTask(self):
        flag = Int32()
        flag.data = self.total_tasks
        self.numOfTasks_pub.publish(flag)
        

if __name__ == "__main__":
    manager = Manager()

    