#!/usr/bin/env python3
#=====================================
#      Simple controller for the        
#          SML Nexus robot
#               ----
#   Listen to the robot pose from
#    the mocap system and output a
#   velocity command to drive the
#    robot back and forth between
#             two points
#=====================================
import sys
import rospy
import copy
import casadi as ca
import numpy as np
import geometry_msgs.msg
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from builders import AlwaysOperator, StlTask, TimeInterval, go_to_goal_predicate_2d, create_barrier_from_task



class Controller():
    #=====================================
    #         Class constructor
    #  Initializes node and subscribers
    #=====================================
    def __init__(self):
        #Initialize node
        rospy.init_node('test_controller')

        #Get tracked object name from parameters
        robot_name = rospy.get_param('~robot_name')

        #Init robot pose
        self.robot_pose = geometry_msgs.msg.PoseStamped()

        #Init last received pose time
        self.last_received_pose = rospy.Time()

        #Init last received pose time
        self.last_received_tracked_pose = rospy.Time()

        #Timeout in seconds
        timeout = 0.5

        #Setup robot pose subscriber
        rospy.Subscriber("/qualisys/"+robot_name+"/pose", geometry_msgs.msg.PoseStamped, self.pose_callback)


        #Setup velocity command publisher
        vel_pub = rospy.Publisher("cmd_vel", geometry_msgs.msg.Twist, queue_size=100)

        #Setup transform subscriber
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        #Wait for transform to be available
        while not tf_buffer.can_transform('mocap', robot_name, rospy.Time()):
            rospy.loginfo("Wait for transform to be available")
            rospy.sleep(1)

        #Create a ROS Twist message for velocity command
        vel_cmd_msg = geometry_msgs.msg.Twist()


        #-----------------------------------------------------------------
        # Loop at set frequency and publish position command if necessary
        #-----------------------------------------------------------------
        #loop frequency in Hz
        loop_frequency = 50
        r = rospy.Rate(loop_frequency)


        x_sym = ca.MX.sym('state', 2)
        t_sym = ca.MX.sym('time', 1)
        A = ca.DM.zeros(1, 2)  
        b = ca.DM.zeros(1)
        Q = ca.DM_eye(2)

        #======================================================
        # Temporary code to test the barrier function
        #======================================================
        
        #----------- Will get thesee by subscribing to a topic -----
        initial_state = {1:np.array([0,0])} #This might even be initialized in the launch file
        goal = np.array([5,4])
        a = 20
        b = 55
        #------------------------------------------------------------

        predicate = go_to_goal_predicate_2d(goal=goal,epsilon = 1, position1=x_sym) # Don't have DynamicalModel
        always    = AlwaysOperator(time_interval=TimeInterval(a=a,b=b)) 
        task      = StlTask(predicate=predicate,temporal_operator=always)

        scale_factor    = 3
        dummy_scalar    = ca.MX.sym('dummy_scalar',1)
        alpha_fun  = ca.Function('alpha_fun',[dummy_scalar],[scale_factor*dummy_scalar]) 

        barrier = create_barrier_from_task(task=task,initial_conditions=initial_state,alpha_function=alpha_fun)

        #======================================================
        input_values = {}
        

        while not rospy.is_shutdown():

            t  = rospy.Time.now().to_sec()
            
            if (t < self.last_received_pose.to_sec() + timeout):
                x = ca.vertcat(self.robot_pose.pose.position.x, self.robot_pose.pose.position.y)
                input_values['state_1'] = x
                input_values['time'] = t
                
                # Compute the barrier function value and its gradients
                barrier_val = barrier.function.call(input_values)['value']
                A = barrier.gradient_function_wrt_state_of_agent(1).call(input_values)['value'] # Change so it works with multiple agents
                b = barrier.partial_time_derivative.call(input_values)['value'] + barrier_val

                if np.linalg.norm(A) < 1e-10:
                    u_hat = ca.MX([0, 0])
                else:
                    # Defining the problem and the inequality constraints
                    u_hat = ca.MX.sym('u_hat', 2)
                    objective = u_hat.T @ Q @ u_hat
                    constraint = A @ u_hat + b 
                    qp = {'x':u_hat, 'f':objective, 'g':constraint}

                    # Solving the quadratic program
                    qpsolver = ca.qpsol('u_opt', 'qpoases', qp) 
                    u_opt = qpsolver(lbg=0)

                    # Extract control input
                    u_hat = u_opt['x']

                vel_cmd_msg.linear.x = u_hat[0]
                vel_cmd_msg.linear.y = u_hat[1]        
            else:
                vel_cmd_msg.linear.x = 0
                vel_cmd_msg.linear.y = 0
            
            #-----------------
            # Publish command
            #-----------------
            try:
                #Get transform from mocap frame to robot frame
                transform = tf_buffer.lookup_transform('mocap', robot_name, rospy.Time())
                #
                vel_cmd_msg_transformed = transform_twist(vel_cmd_msg, transform)
                #Publish cmd message
                vel_pub.publish(vel_cmd_msg_transformed)
            except:
                continue

            #---------------------------------
            # Sleep to respect loop frequency
            #---------------------------------
            r.sleep()

    #=====================================
    #          Callback function 
    #      for robot pose feedback
    #=====================================
    def pose_callback(self, pose_stamped_msg):
        #Save robot pose as class variable
        self.robot_pose = pose_stamped_msg

        #Save when last pose was received
        self.last_received_pose = rospy.Time.now()




#=====================================
# Apply transform to a twist message 
#     including angular velocity
#=====================================
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

    #Populate new twist message
    new_twist = geometry_msgs.msg.Twist()
    new_twist.linear = out_vel.vector
    new_twist.angular = out_rot.vector

    return new_twist

#=====================================
#               Main
#=====================================
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
