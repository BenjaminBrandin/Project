#!/usr/bin/env python3
import sys
import rospy
import copy
from builders import BarrierFunction
from typing import List, Dict
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
# import geometry_msgs.msg
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Vector3Stamped
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from simulation_graph import barriers, initial_states
 

class Controller():

    def __init__(self):

        rospy.init_node('controller')

        self.Q = ca.DM_eye(2)
        self.solver = ca.Function = None
        self.input_vector = ca.MX.sym('input', 2)
        self.vel_cmd_msg = Twist()
        self.parameters = ca_tools.structure3.msymStruct = None

        self.agent_pose = PoseStamped()
        self.agent_name = rospy.get_param('~robot_name')
        self.agent_id = int(self.agent_name[-1])
        self.last_received_pose = rospy.Time()

        self.barriers = barriers
        self.barrier_constraints: ca.MX = None
        self.initial_states = initial_states
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        # self.relevant_barriers = [self.barriers[self.agent_id - 1]]

        self.agents = [agent for agent in self.initial_states.keys()]
        self.neighbour_agents: Dict[int, PoseStamped] = {}


        # Setup subscriber for agent pose
        rospy.Subscriber("/qualisys/"+self.agent_name+"/pose", PoseStamped, self.pose_callback)

        # Setup subscribers for all other agents
        for id in self.agents:
            if id != self.agent_id:
                # rospy.Subscriber(f"/qualisys/nexus{id}/pose", PoseStamped, self.other_agent_pose_callback)
                rospy.Subscriber(f"/nexus{id}/agent_pose", PoseStamped, self.other_agent_pose_callback)
            else:
                continue

        # Setup publisher for velocity commands and agent pose
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=100)
        self.agent_pose_pub = rospy.Publisher(f"agent_pose", PoseStamped, queue_size=100)
            

        #Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.tf_buffer.can_transform('mocap', self.agent_name, rospy.Time()):
            rospy.loginfo("Wait for transform to be available")
            rospy.sleep(1)

        self.solver = self.set_up_optimization_problem()
        self.control_loop()


    
    def set_up_optimization_problem(self):
        # Create the structure for the optimization problem
        parameter_list = []
        parameter_list += [ca_tools.entry(f"state_{self.agent_id}", shape=2)]
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in self.agents if id != self.agent_id] 
        parameter_list += [ca_tools.entry("time", shape=1)]
        self.parameters = ca_tools.struct_symMX(parameter_list)

        # Create the barrier constraints and the objective function
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



    def pose_callback(self, msg):
        self.agent_pose = msg
        self.last_received_pose = rospy.Time.now()
        self.agent_pose_pub.publish(self.agent_pose)

    def other_agent_pose_callback(self, msg):
        agent_id = int(msg._connection_header['topic'].split('/')[-2].replace('nexus', '')) 
        self.neighbour_agents[agent_id] = msg
        # pass


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