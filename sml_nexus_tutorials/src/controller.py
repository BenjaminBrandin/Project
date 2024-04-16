#!/usr/bin/env python3
import sys
import rospy
import copy
from builders import BarrierFunction
from typing import List
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
import geometry_msgs.msg
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from simulation_graph import barriers
 

class Controller():

    def __init__(self):
        #Initialize node
        rospy.init_node('controller')

        self.agent_pose = geometry_msgs.msg.PoseStamped()
        self.tracked_object_pose = geometry_msgs.msg.PoseStamped()
        self.last_received_pose = rospy.Time()
        self.vel_cmd_msg = geometry_msgs.msg.Twist()
        self.parameters = ca_tools.structure3.msymStruct = None
        # self.relevant_barriers = [barrier for barrier in barriers.values() if self.agent_id in barrier.contributing_agents]
        self.solver = ca.Function = None
        self.agent_name = rospy.get_param('~robot_name')
        self.agent_id = int(self.agent_name[-1])
        self.relevant_barriers = barriers[self.agent_id]
        self.barrier_list = [barrier for barrier in barriers.values()]
        self.start_sol = np.array([]) # warm start solution for the optimization problem
        self.input_vector = ca.MX.sym('input', 2)



        rospy.Subscriber("/qualisys/"+self.agent_name+"/pose", geometry_msgs.msg.PoseStamped, self.pose_callback)
        if self.agent_name in ["nexus2", "nexus3"]:
            tracked_object = rospy.get_param('~tracked_object')
            self.tracked_id = int(tracked_object[-1])
            rospy.Subscriber("/qualisys/"+tracked_object+"/pose", geometry_msgs.msg.PoseStamped, self.tracked_object_pose_callback)

        # Setup publisher
        self.vel_pub = rospy.Publisher("cmd_vel", geometry_msgs.msg.Twist, queue_size=100)


        #Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.tf_buffer.can_transform('mocap', self.agent_name, rospy.Time()):
            rospy.loginfo("Wait for transform to be available")
            rospy.sleep(1)

        self.solver = self.set_up_optimization_problem()
        self.control_loop()





    
    def set_up_optimization_problem(self):
        parameter_list = []
        parameter_list += [ca_tools.entry(f"state_{self.agent_id}", shape=2)]
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in barriers.keys() if id != self.agent_id] # Find new way to add the states of the other agents
        parameter_list += [ca_tools.entry("time", shape=1)]
        self.parameters = ca_tools.struct_symMX(parameter_list)

        Q = ca.DM_eye(2)
        barrier_constraint = self.generate_barrier_constraints(self.barrier_list)
        constraint = ca.vertcat(barrier_constraint)
        objective = self.input_vector.T @ Q @ self.input_vector

        qp = {'x': self.input_vector, 'f': objective, 'g': constraint, 'p': self.parameters}
        opts = {'printLevel': 'none'}
        self.qpsolver = ca.qpsol('sol', 'qpoases', qp, opts)

        return self.qpsolver





    def generate_barrier_constraints(self,barrier_list:List[BarrierFunction]) -> ca.MX:

        constraints = []
        for barrier in barrier_list:

            named_inputs = {"state_"+str(id):self.parameters["state_"+str(id)] for id in barrier.contributing_agents}
            named_inputs["time"] = self.parameters["time"]
            
            nabla_xi_fun = barrier.gradient_function_wrt_state_of_agent(self.agent_id) 
            partial_time_derivative_fun = barrier.partial_time_derivative
            barrier_fun = barrier.function

            # just evaluate to get the symbolic expression now
            nabla_xi = nabla_xi_fun.call(named_inputs)["value"] # symbolic expression of the gradient of the barrier function w.r.t to the state of the agent
            dbdt     = partial_time_derivative_fun.call(named_inputs)["value"] # symbolic expression of the partial time derivative of the barrier function
            alpha_b  = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"]) # symbolic expression of the barrier function

            if self.agent_id == 1:
                load_sharing = 1
            else:
                load_sharing = 0.5

            barrier_constraint   = -1* ( ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b))
            constraints += [barrier_constraint] 
                

        return ca.vertcat(*constraints)



    def control_loop(self):
        r = rospy.Rate(50)
        while not rospy.is_shutdown():

            # ------ Fill the structure with the current values ------
            current_parameters = self.parameters(0)
            t  = rospy.Time.now().to_sec()
            current_parameters["time"] = t
  
            state = ca.vertcat(self.agent_pose.pose.position.x, self.agent_pose.pose.position.y)
            current_parameters[f'state_{self.agent_id}'] = state

            if self.agent_name in ["nexus2", "nexus3"]:
                tracked_state = ca.vertcat(self.tracked_object_pose.pose.position.x, self.tracked_object_pose.pose.position.y)
                current_parameters[f'state_{self.tracked_id}'] = tracked_state

            # --------------------------------------------------------

            sol = self.solver(p = current_parameters, ubg = 0)  # Might have made a mistake in defining the objective function or the constraints.
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

                


    def pose_callback(self, pose_stamped_msg):
        self.agent_pose = pose_stamped_msg
        self.last_received_pose = rospy.Time.now()

    def tracked_object_pose_callback(self, pose_stamped_msg):
        self.tracked_object_pose = pose_stamped_msg
        self.last_received_tracked_pose = rospy.Time.now()


def transform_twist(twist = geometry_msgs.msg.Twist, transform_stamped = geometry_msgs.msg.TransformStamped):

    transform_stamped_ = copy.deepcopy(transform_stamped)
    #Inverse real-part of quaternion to inverse rotation
    transform_stamped_.transform.rotation.w = - transform_stamped_.transform.rotation.w

    twist_vel = geometry_msgs.msg.Vector3Stamped()
    twist_rot = geometry_msgs.msg.Vector3Stamped()
    twist_vel.vector = twist.linear
    twist_rot.vector = twist.angular
    out_vel = tf2_geometry_msgs.do_transform_vector3(twist_vel, transform_stamped_)
    out_rot = tf2_geometry_msgs.do_transform_vector3(twist_rot, transform_stamped_)

    new_twist = geometry_msgs.msg.Twist()
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