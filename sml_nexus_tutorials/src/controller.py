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
from simulation_graph import barriers_dict
 

class Controller():

    def __init__(self):
        #Initialize node
        rospy.init_node('controller')

        self.agent_pose = geometry_msgs.msg.PoseStamped()
        self.tracked_object_pose = geometry_msgs.msg.PoseStamped()
        self.last_received_pose = rospy.Time()
        self.vel_cmd_msg = geometry_msgs.msg.Twist()
        self.parameters = ca_tools.structure3.msymStruct = None
        # self.relevant_barriers = [barrier for barrier in barriers_dict.values() if self.agent_id in barrier.contributing_agents]
        self.solver = ca.Function = None
        self.agent_name = rospy.get_param('~robot_name')
        self.agent_id = int(self.agent_name[-1])
        self.relevant_barriers = barriers_dict[self.agent_id]
        self.start_sol = np.array([]) # warm start solution for the optimization problem

        self.state_vector = ca.MX.sym('state', 2)
        self.input_vector = ca.MX.sym('input', 2)
        self.dynamic_model = ca.Function('dynamics',[self.state_vector,self.input_vector],[self.input_vector],["state","input"],["value"])
        


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
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in barriers_dict.keys() if id != self.agent_id] # Find new way to add the states of the other agents
        parameter_list += [ca_tools.entry("time", shape=1)]
        # parameter_list += [ca_tools.entry("A", shape=(1,2))]
        # parameter_list += [ca_tools.entry("b", shape=1)]

        self.parameters = ca_tools.struct_symMX(parameter_list)



        A = ca.MX.sym('A', 1, 2)
        b = ca.MX.sym('b', 1)
        Q = ca.DM_eye(2)


        input_constraint = A @ self.input_vector - b
        barrier_constraint = self.generate_barrier_constraints(self.relevant_barriers) # Wrong barriers
        constraint = ca.vertcat(input_constraint, barrier_constraint)
        objective = self.input_vector.T @ Q @ self.input_vector


        qp = {'x': self.input_vector, 'f': objective, 'g': constraint, 'p': self.parameters}
        opts = {'printLevel': 'none'}
        self.qpsolver = ca.qpsol('sol', 'qpoases', qp, opts)

        return self.qpsolver



    def generate_barrier_constraints(self,barriers:List[BarrierFunction]) -> ca.MX:

        constraints = []
        for barrier in barriers :

            named_inputs = {"state_"+str(unique_identifier):self.parameters["state_"+str(unique_identifier)]  for unique_identifier in barrier.contributing_agents}
            named_inputs["time"] = self.parameters["time"] # add the time  
            
            nabla_xi_fun                : ca.Function   = barrier.gradient_function_wrt_state_of_agent(self.agent_id) # this will have the the same inputs as the barrier itself
            partial_time_derivative_fun : ca.Function   = barrier.partial_time_derivative
            barrier_fun                 : ca.Function   = barrier.function
            dyn                         : ca.Function   = self.dynamic_model
            
            self._nabla_xi_fun = nabla_xi_fun

            # just evaluate to get the symbolic expression now
            nabla_xi = nabla_xi_fun.call(named_inputs)["value"] # symbolic expression of the gradient of the barrier function w.r.t to the state of the agent
            dbdt     = partial_time_derivative_fun.call(named_inputs)["value"] # symbolic expression of the partial time derivative of the barrier function
            alpha_b  = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"]) # symbolic expression of the barrier function
            dynamics = dyn.call({"state":self.parameters["state_"+str(self.agent_id)],"input":self.input_vector})["value"] # symbolic expression of the dynamics of the agent
            
            load_sharing      = 0.5
            barrier_constraint   = -1* ( ca.dot(nabla_xi.T, dynamics ) + load_sharing * (dbdt + alpha_b))
            constraints += [barrier_constraint] # add constraints to the list of constraints
                

        return ca.vertcat(*constraints)



    def control_loop(self):
        timeout = 0.5
        # input_values = {}
        loop_frequency = 50
        r = rospy.Rate(loop_frequency)

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

            # if self.start_sol.size != 0:
            #     try :
            #         sol = self.solver(x0 = self.start_sol, p = current_parameters, ubg = 0)
            #         self.start_sol = sol['x'] 
            #     except Exception as e1:
            #         print(f"Agent {self.agent_id} Primary Controller Failed !")
            #         self._logger.error(f"Primary controller failed with the following message")
            #         self._logger.error(e1, exc_info=True)  
            #         raise e1
                      
            # else :
            #     try :
            #         sol = self.solver(p = current_parameters, ubg = 0)
            #         self.start_sol = sol['x'] 
                    
            #     except Exception as e1:
            #         print(f"Agent {self.agent_id} Primary Controller Failed !")
            #         self._logger.error(f"Primary controller failed with the following message")
            #         self._logger.error(e1, exc_info=True)
            #         raise e1

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