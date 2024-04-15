#!/usr/bin/env python3
import sys
import rospy
import copy
import casadi as ca
import numpy as np
import geometry_msgs.msg
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
# from simulation import barriers
from simulation_graph import barriers
 

class Controller():

    def __init__(self):
        #Initialize node
        rospy.init_node('test_controller')

        self.robot_pose = geometry_msgs.msg.PoseStamped()
        self.tracked_object_pose = geometry_msgs.msg.PoseStamped()
        self.last_received_pose = rospy.Time()
        vel_cmd_msg = geometry_msgs.msg.Twist()

        # Initialize a structure line 618
 
        # ------Find a better way to do this------
        # Get robot name from parameter server
        robot_name = rospy.get_param('~robot_name')
        robot_id = int(robot_name[-1])
        rospy.Subscriber("/qualisys/"+robot_name+"/pose", geometry_msgs.msg.PoseStamped, self.pose_callback)

        # Get tracked object name from parameter server
        if robot_name == "nexus2" or robot_name == "nexus3":
            tracked_object = rospy.get_param('~tracked_object')
            tracked_id = int(tracked_object[-1])
            rospy.Subscriber("/qualisys/"+tracked_object+"/pose", geometry_msgs.msg.PoseStamped, self.tracked_object_pose_callback)
        else:
            pass
        #  ----------------------------------------

        # Setup publisher
        vel_pub = rospy.Publisher("cmd_vel", geometry_msgs.msg.Twist, queue_size=100)


        #Setup transform subscriber
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        while not tf_buffer.can_transform('mocap', robot_name, rospy.Time()):
            rospy.loginfo("Wait for transform to be available")
            rospy.sleep(1)


        timeout = 0.5
        input_values = {}
        loop_frequency = 50
        r = rospy.Rate(loop_frequency)
        relevant_barriers = barriers[robot_id]
        # relevant_barriers = [barrier for barrier in barriers.values() if robot_id in barrier.contributing_agents]
        

        # Define the optimization problem parameters
        u_hat = ca.MX.sym('u_hat', 2)
        x_sym = ca.MX.sym('state', 2)
        t_sym = ca.MX.sym('time', 1)
        A = ca.MX.sym('A', 1, 2)   
        b = ca.MX.sym('b', 1)
        Q = ca.DM_eye(2)
        
        constraint = A @ u_hat + b 
        objective = u_hat.T @ Q @ u_hat
        params = ca.vertcat(x_sym, t_sym, A.T, b)
        qp = {'x': u_hat, 'f': objective, 'g': constraint, 'p': params}  # g will be created with the function generate_barrier_constraints line 718
        opts = {'printLevel': 'none'}
        qpsolver = ca.qpsol('u_opt', 'qpoases', qp, opts)
        

        while not rospy.is_shutdown():

            t  = rospy.Time.now().to_sec()
            
            if (t < self.last_received_pose.to_sec() + timeout):
                
                # ------Find a better way to do this------
                state = ca.vertcat(self.robot_pose.pose.position.x, self.robot_pose.pose.position.y)
                input_values["time"] = t
                input_values[f'state_{robot_id}'] = state

                if robot_name == "nexus2" or robot_name == "nexus3": 
                    input_values[f'state_{tracked_id}'] = ca.vertcat(self.tracked_object_pose.pose.position.x, self.tracked_object_pose.pose.position.y)
                else:
                    pass
                #  ----------------------------------------

                barrier_val = relevant_barriers.function.call(input_values)['value']
                A_val = relevant_barriers.gradient_function_wrt_state_of_agent(robot_id).call(input_values)['value'] 
                b_val = (relevant_barriers.partial_time_derivative.call(input_values)['value'] + barrier_val)/(len(input_values)-1)  # Divide by the number of agents

                

                if np.linalg.norm(A_val) < 1e-10:
                    u_hat = ca.MX([0, 0])
                else:
                    param_values = ca.vertcat(state, t, A_val.T, b_val) 
                    u_opt = qpsolver(lbg=0, p=param_values) # function to solve the optimization problem that first set zero to all in the structure
                    u_hat = u_opt['x']

                vel_cmd_msg.linear.x = u_hat[0]
                vel_cmd_msg.linear.y = u_hat[1]        
            else:
                vel_cmd_msg.linear.x = 0
                vel_cmd_msg.linear.y = 0
            
            try:
                # Get transform from mocap frame to robot frame
                transform = tf_buffer.lookup_transform('mocap', robot_name, rospy.Time())
                vel_cmd_msg_transformed = transform_twist(vel_cmd_msg, transform)
                vel_pub.publish(vel_cmd_msg_transformed)
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
