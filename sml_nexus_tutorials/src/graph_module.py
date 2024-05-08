"""
MIT License

Copyright (c) [2024] [Gregorio Marchesini]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import networkx as nx 
import numpy   as np
import casadi  as ca
from builders import StlTask,TimeInterval, AlwaysOperator, epsilon_position_closeness_predicate 

from typing import Tuple, List, Dict, Union
UniqueIdentifier = int



from   dataclasses import dataclass,field

# recall that mutable objects remain mutable even of the class is frozen
@dataclass(unsafe_hash=True)
class EdgeTaskContainer:
    _edge_tuple: Tuple[UniqueIdentifier,UniqueIdentifier]
    _weight: float = 1
    _task_list: List[StlTask] = field(default_factory=list)
    _involved_in_optimization: int = 0


    def __init__(self, edge_tuple: Tuple[UniqueIdentifier,UniqueIdentifier] = None, weight: float = None, task_list: List[StlTask] = None, involved_in_optimization: int = None):

        self._edge_tuple = edge_tuple if edge_tuple is not None else (UniqueIdentifier(), UniqueIdentifier())
        self._weight = weight if weight is not None else 1
        self._task_list = task_list if task_list is not None else []
        self._involved_in_optimization = involved_in_optimization if involved_in_optimization is not None else 0


    @property
    def edge_tuple(self):
        return self._edge_tuple
    @property
    def weight(self):
        return self._weight
    @property
    def task_list(self):
        return self._task_list
    @property
    def involved_in_optimization(self) :
      return self._involved_in_optimization
      
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Weight should be a positive number")
        for task in self.task_list:
            if not isinstance(task, StlTask):
                raise ValueError("The tasks list should contain only StlTask objects")
        
        
    def _add_single_task(self, input_task: StlTask) -> None:
        """ Set the tasks for the edge that has to be respected by the edge. Input is expected to be a list  """
       
        if not isinstance(input_task, StlTask):
            raise Exception("please enter a valid STL task object or a list of StlTask objects")
        else:
            if isinstance(input_task, StlTask):
                # set the source node pairs of this node
                nodei,nodej = self.edge_tuple

                if (not nodei in input_task.contributing_agents) or (not nodej in input_task.contributing_agents) :
                    raise Exception(f"the task {input_task} is not compatible with the edge {self.edge_tuple}. The contributing agents are {input_task.contributing_agents}")
                else:
                    self._task_list.append(input_task) # adding a single task
            
    # check task addition
    def add_tasks(self, tasks: Union[StlTask, List[StlTask]]):
        if isinstance(tasks, list): # list of tasks
            for task in tasks:
                self._add_single_task(task)
        else: # single task case
            self._add_single_task(tasks)
    
    def cleanTasks(self)-> None :
        self.task_list      = []

    def flagOptimizationInvolvement(self) -> None :
        self._involved_in_optimization = 1
    
        
def create_communication_graph_from_states(states: List[int], communication_info: Dict[str, Dict]) -> nx.Graph :
    comm_graph = nx.Graph()
    comm_graph.add_nodes_from(states)

    # Add self-loops
    for node in states:
        comm_graph.add_edge(node, node)

    for task_info in communication_info.values():
        if task_info["COMMUNICATE"]:
            edge = task_info["EDGE"]
            if edge[0] in states and edge[1] in states:
                comm_graph.add_edge(edge[0],edge[1])
            else:
                raise Exception(f"Edge {edge} is not compatible with the states {states}")

    return comm_graph


def create_task_graph_from_edges(edge_list: Union[List[EdgeTaskContainer], List[Tuple[UniqueIdentifier, UniqueIdentifier]]]) -> nx.Graph:

    task_graph = nx.Graph()
    for edge in edge_list:
        task_graph.add_edge(edge[0], edge[1], container=EdgeTaskContainer(edge_tuple=edge))
    return task_graph

