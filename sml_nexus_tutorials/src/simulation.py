#!/usr/bin/env python3
import numpy as np
import casadi as ca
from builders import AlwaysOperator, StlTask, TimeInterval, go_to_goal_predicate_2d, create_barrier_from_task

symbolic_state = ca.MX.sym('state', 2)
barriers = {}
# in the graph the vertices are the agents and the edges are the tasks
tasks = [
    {"initial_state": np.array([0, 0]), "goal": np.array([6, 6])},
    {"initial_state": np.array([0, -2]), "goal": np.array([0, 8])}
]



for i, task_info in enumerate(tasks, start=1):
    initial_state = {1: task_info["initial_state"]}
    goal = task_info["goal"]

    predicate = go_to_goal_predicate_2d(goal=goal, epsilon=1, position1=symbolic_state)
    always = AlwaysOperator(time_interval=TimeInterval(a=20, b=55))
    task = StlTask(predicate=predicate, temporal_operator=always)

    scale_factor = 3
    dummy_scalar = ca.MX.sym('dummy_scalar', 1)
    alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

    # Before creating the barrier, we need to decompose the tasks that are not in the communication graph

    barrier = create_barrier_from_task(task=task, initial_conditions=initial_state, alpha_function=alpha_fun)
    barriers[i] = barrier
