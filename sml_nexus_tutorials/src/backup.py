#!/usr/bin/env python3
import sys
import rospy
import copy
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

        self.robot_pose = geometry_msgs.msg.PoseStamped()
        self.tracked_object_pose = geometry_msgs.msg.PoseStamped()
        self.last_received_pose = rospy.Time()
        self.vel_cmd_msg = geometry_msgs.msg.Twist()
        self.parameters = ca_tools.structure3.msymStruct = None
        # self.relevant_barriers = [barrier for barrier in barriers.values() if self.robot_id in barrier.contributing_agents]
        self.solver = ca.Function = None
        self.robot_name = rospy.get_param('~robot_name')
        self.robot_id = int(self.robot_name[-1])
        self.relevant_barriers = barriers[self.robot_id]


        rospy.Subscriber("/qualisys/"+self.robot_name+"/pose", geometry_msgs.msg.PoseStamped, self.pose_callback)
        if self.robot_name in ["nexus2", "nexus3"]:
            tracked_object = rospy.get_param('~tracked_object')
            self.tracked_id = int(tracked_object[-1])
            rospy.Subscriber("/qualisys/"+tracked_object+"/pose", geometry_msgs.msg.PoseStamped, self.tracked_object_pose_callback)

        # Setup publisher
        self.vel_pub = rospy.Publisher("cmd_vel", geometry_msgs.msg.Twist, queue_size=100)


        #Setup transform subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not self.tf_buffer.can_transform('mocap', self.robot_name, rospy.Time()):
            rospy.loginfo("Wait for transform to be available")
            rospy.sleep(1)

        self.solver = self.set_up_optimization_problem()
        self.control_loop()



    

    def set_up_optimization_problem(self):
        self.input_vector = ca.MX.sym('input', 2)
        self.Q = ca.DM_eye(2)

        parameter_list = []
        parameter_list += [ca_tools.entry(f"state_{self.robot_id}", shape=2)]
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in barriers.keys() if id != self.robot_id]
        parameter_list += [ca_tools.entry("time", shape=1)]
        # Initialization failed since variables [A, b] are free. These symbols occur in the output expressions but you forgot to declare these as inputs.
        parameter_list += [ca_tools.entry("A", shape=(1,2))]
        parameter_list += [ca_tools.entry("b", shape=1)]
        
        self.parameters = ca_tools.struct_symMX(parameter_list)

        constraint = self.parameters["A"] @ self.input_vector + self.parameters["b"] 
        objective = self.input_vector.T @ self.Q @ self.input_vector


        qp = {'x': self.input_vector, 'f': objective, 'g': constraint, 'p': self.parameters}
        opts = {'printLevel': 'none'}
        self.qpsolver = ca.qpsol('sol', 'qpoases', qp, opts)

        return self.qpsolver


    def control_loop(self):
        timeout = 0.5
        input_values = {}
        loop_frequency = 50
        r = rospy.Rate(loop_frequency)

        while not rospy.is_shutdown():
            current_parameters = self.parameters(0)

            t  = rospy.Time.now().to_sec()
            current_parameters["time"] = t
            
            if (t < self.last_received_pose.to_sec() + timeout):
                
                # ------Find a better way to do this------
                state = ca.vertcat(self.robot_pose.pose.position.x, self.robot_pose.pose.position.y)
                current_parameters[f'state_{self.robot_id}'] = state
                input_values["time"] = t
                input_values[f'state_{self.robot_id}'] = state

                if self.robot_name in ["nexus2", "nexus3"]:
                    tracked_state = ca.vertcat(self.tracked_object_pose.pose.position.x, self.tracked_object_pose.pose.position.y)
                    input_values[f'state_{self.tracked_id}'] = tracked_state
                    current_parameters[f'state_{self.tracked_id}'] = tracked_state
                else:
                    pass
                #  ----------------------------------------

                barrier_val = self.relevant_barriers.function.call(input_values)['value']
                A_val = self.relevant_barriers.gradient_function_wrt_state_of_agent(self.robot_id).call(input_values)['value'] 
                b_val = (self.relevant_barriers.partial_time_derivative.call(input_values)['value'] + barrier_val)/(len(input_values)-1)  # Divide by the number of agents
                current_parameters["A"] = A_val
                current_parameters["b"] = b_val
                

                if np.linalg.norm(A_val) < 1e-10:
                    input = ca.MX([0, 0])
                else:
                    sol = self.solver(p=current_parameters, lbg=0)
                    input = sol['x']
                    const = sol['g']
                    rospy.loginfo(f"Constraint: {const}")

                self.vel_cmd_msg.linear.x = input[0]
                self.vel_cmd_msg.linear.y = input[1]        
            else:
                self.vel_cmd_msg.linear.x = 0
                self.vel_cmd_msg.linear.y = 0
            
            try:
                # Get transform from mocap frame to robot frame
                transform = self.tf_buffer.lookup_transform('mocap', self.robot_name, rospy.Time())
                vel_cmd_msg_transformed = transform_twist(self.vel_cmd_msg, transform)
                self.vel_pub.publish(vel_cmd_msg_transformed)
            except:
                continue

            r.sleep()

    
    def pose_callback(self, pose_stamped_msg):
        self.robot_pose = pose_stamped_msg
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