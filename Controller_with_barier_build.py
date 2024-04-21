#!/usr/bin/env python3
import sys
import rospy
import copy
from builders import BarrierFunction, Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, go_to_goal_predicate_2d, formation_predicate, create_barrier_from_task
from graph_module import create_communication_graph_from_states, create_task_graph_from_edges
from typing import List, Dict
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Vector3Stamped
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
# from simulation_graph import barriers, initial_states

class Controller():

    def __init__(self):

        # Initialize the node
        rospy.init_node('controller')

        # Optimization Problem
        self.solver = None
        self.Q = ca.DM_eye(2)
        self.parameters = None
        self.barriers = []
        self.barrier_constraints = None
        self.input_vector = ca.MX.sym('input', 2)
        
        # Agent Information
        self.agent_pose = PoseStamped()
        self.agent_name = rospy.get_param('~robot_name')
        self.agent_id = int(self.agent_name[-1])
        self.last_received_pose = rospy.Time()

        # Neighbouring Agents
        self.initial_states = {}
        self.agents = []
        self.neighbour_agents = {}
        
        # Velocity Command Message
        self.vel_cmd_msg = Twist()

    
        # Setup subscribers
        rospy.Subscriber("/qualisys/"+self.agent_name+"/pose", PoseStamped, self.pose_callback)

        for id in self.agents:
            if id != self.agent_id:
                rospy.Subscriber(f"/nexus{id}/agent_pose", PoseStamped, self.other_agent_pose_callback)
            else:
                continue

        # Setup publishers
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=100)
        self.agent_pose_pub = rospy.Publisher(f"agent_pose", PoseStamped, queue_size=100)
            
        #Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.tf_buffer.can_transform('mocap', self.agent_name, rospy.Time()):
            rospy.loginfo("Wait for transform to be available")
            rospy.sleep(1)

        self.barriers, self.initial_states = self.set_up_graphs_and_barriers()
        self.agents = list(self.initial_states.keys()) 
        self.solver = self.set_up_optimization_problem()
        self.control_loop()


    def set_up_graphs_and_barriers(self):
        barriers = []
        initial_states = {}
        edge_barriers = []
        scale_factor = 3
        communication_radius = 3.0

        # Initial states of the robots 
        state1 = np.array([0,0]) 
        state2 = np.array([0,-2])    
        state3 = np.array([0,2])     
        # initial_states = {1:state1,2:state2,3:state3}
        initial_states = {1:state1,2:state2}
         

        # Creating the robots
        robot_1 = Agent(id=1, initial_state=state1)
        robot_2 = Agent(id=2, initial_state=state2)
        robot_3 = Agent(id=3, initial_state=state3)

        # Creating the graphs
        # task_edges = [(1,1),(1,2),(1,3)]
        task_edges = [(1,1),(1,2)]
        task_graph = create_task_graph_from_edges(edge_list = task_edges) # creates an empty task graph
        comm_graph = create_communication_graph_from_states(initial_states,communication_radius)

        # Creating the alpha function that is the same for all the tasks for now
        dummy_scalar = ca.MX.sym('dummy_scalar', 1)
        alpha_fun = ca.Function('alpha_fun', [dummy_scalar], [scale_factor * dummy_scalar])

        # ============ Task 1 ====================================================================================================
        edge_1 = task_graph[1][1]["container"]
        predicate = go_to_goal_predicate_2d(goal=np.array([6, 2]), epsilon=3, agent=robot_1)
        temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=20, b=50))
        task = StlTask(predicate=predicate, temporal_operator=temporal_operator)
        barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1], alpha_function=alpha_fun)) 
        edge_1.add_tasks(task)
        # =========================================================================================================================


        # ============ Task 2 =====================================================================================================
        edge_12 = task_graph[1][2]["container"]
        predicate = formation_predicate(epsilon=2, agent_i = robot_1, agent_j = robot_2, relative_pos=np.array([-1,-1]))
        temporal_operator = EventuallyOperator(time_interval=TimeInterval(a=30,b=40)) 
        task = StlTask(predicate=predicate,temporal_operator=temporal_operator)
        barriers.append(create_barrier_from_task(task=task, initial_conditions=[robot_1, robot_2], alpha_function=alpha_fun)) 
        edge_12.add_tasks(task)
        # =========================================================================================================================
        return barriers, initial_states

    
    def set_up_optimization_problem(self):
        # Create the structure for the optimization problem
        parameter_list = []
        parameter_list += [ca_tools.entry(f"state_{self.agent_id}", shape=2)]
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in self.agents if id != self.agent_id] 
        parameter_list += [ca_tools.entry("time", shape=1)]
        self.parameters = ca_tools.struct_symMX(parameter_list)

        # Create the barrier constraints and the objective function
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        self.barrier_constraints = self.generate_barrier_constraints(self.relevant_barriers)
        constraint = ca.vertcat(self.barrier_constraints)
        objective = self.input_vector.T @ self.Q @ self.input_vector

        # Create the optimization solver
        qp = {'x': self.input_vector, 'f': objective, 'g': constraint, 'p': self.parameters}
        opts = {'printLevel': 'none'}
        self.qpsolver = ca.qpsol('sol', 'qpoases', qp, opts)

        return self.qpsolver



    def generate_barrier_constraints(self, barrier_list:List[BarrierFunction]) -> ca.MX:

        constraints = []
        for barrier in barrier_list:

            named_inputs = {"state_"+str(id):self.parameters["state_"+str(id)] for id in barrier.contributing_agents}
            named_inputs["time"] = self.parameters["time"]

            nabla_xi_fun = barrier.gradient_function_wrt_state_of_agent(self.agent_id)
            partial_time_derivative_fun = barrier.partial_time_derivative
            barrier_fun = barrier.function

            nabla_xi = nabla_xi_fun.call(named_inputs)["value"]                                        
            dbdt = partial_time_derivative_fun.call(named_inputs)["value"]                             
            alpha_b = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"])       

            # Fix load sharing for different tasks
            load_sharing = 1/len(barrier.contributing_agents)
            barrier_constraint   = -1* ( ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b))
            constraints.append(barrier_constraint) 
  
        return ca.vertcat(*constraints)



    def control_loop(self):
        r = rospy.Rate(50)
        while not rospy.is_shutdown():

            # Fill the structure with the current values
            current_parameters = self.parameters(0)
            current_parameters["time"] = ca.vertcat(self.last_received_pose.to_sec())
            current_parameters[f'state_{self.agent_id}'] = ca.vertcat(self.agent_pose.pose.position.x, self.agent_pose.pose.position.y)

            for id in self.neighbour_agents.keys():
                current_parameters[f'state_{id}'] = ca.vertcat(self.neighbour_agents[id].pose.position.x, self.neighbour_agents[id].pose.position.y)

            # Might need to check the norm of the gradient matrix to see if it is zero to set optimal_input to zero 

            # Solve the optimization problem and publish the velocity command 
            sol = self.solver(p = current_parameters, ubg = 0) 
            optimal_input  = sol['x']

            self.vel_cmd_msg.linear.x = optimal_input[0]
            self.vel_cmd_msg.linear.y = optimal_input[1]        

            try:
                # Get transform from mocap frame to agent frame
                transform = self.tf_buffer.lookup_transform('mocap', self.agent_name, rospy.Time())
                vel_cmd_msg_transformed = transform_twist(self.vel_cmd_msg, transform)
                self.vel_pub.publish(vel_cmd_msg_transformed)
            except:
                continue

            r.sleep()


    # ----------- Callbacks -----------------
    
    def pose_callback(self, msg):
        self.agent_pose = msg
        self.last_received_pose = rospy.Time.now()
        self.agent_pose_pub.publish(self.agent_pose)

    def other_agent_pose_callback(self, msg):
        agent_id = int(msg._connection_header['topic'].split('/')[-2].replace('nexus', '')) 
        self.neighbour_agents[agent_id] = msg
        



def transform_twist(twist = Twist, transform_stamped = TransformStamped):

    transform_stamped_ = copy.deepcopy(transform_stamped)
    #Inverse real-part of quaternion to inverse rotation
    transform_stamped_.transform.rotation.w = - transform_stamped_.transform.rotation.w

    twist_vel = Vector3Stamped()
    twist_rot = Vector3Stamped()
    twist_vel.vector = twist.linear
    twist_rot.vector = twist.angular
    out_vel = tf2_geometry_msgs.do_transform_vector3(twist_vel, transform_stamped_)
    out_rot = tf2_geometry_msgs.do_transform_vector3(twist_rot, transform_stamped_)

    new_twist = Twist()
    new_twist.linear = out_vel.vector
    new_twist.angular = out_rot.vector

    return new_twist



if __name__ == "__main__":
    controller = Controller()
    try:
        rospy.spin()
    except ValueError as e:
        rospy.logerr(e)
        sys.exit(0)
    except rospy.ROSInterruptException:
        print("Shutting down")
        sys.exit(0)
