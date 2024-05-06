#!/usr/bin/env python3
import sys
import rospy
import copy
from builders import BarrierFunction, Agent, StlTask, TimeInterval, AlwaysOperator, EventuallyOperator, create_barrier_from_task, go_to_goal_predicate_2d, formation_predicate
from typing import List, Dict
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Vector3Stamped
from custom_msg.msg import task_msg
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
# from manager import agents, barriers


class Controller():

    def __init__(self):

        # Initialize the node
        rospy.init_node('controller')

        # Optimization Problem
        self.solver = None
        self.parameters = None
        self.input_vector = ca.MX.sym('input', 2)
        self.slack_variables = {}
        self.scale_factor = 3
        self.dummy_scalar = ca.MX.sym('dummy_scalar', 1)
        self.alpha_fun = ca.Function('alpha_fun', [self.dummy_scalar], [self.scale_factor * self.dummy_scalar])

        # Nabla variables
        self.nabla_funs = []
        self.nabla_inputs = []
        self.nabla_val = []
        
        # Agent Information
        self.agent_pose = PoseStamped()
        self.agent_name = rospy.get_param('~robot_name')
        self.agent_id = int(self.agent_name[-1])
        self.last_received_pose = rospy.Time()

        # Neighbouring Agents
        self.total_agents = rospy.get_param('~num_robots')
        self.agents = {}
        self.neighbour_agents = {}
        initial_states = [np.array([-3, -3]), np.array([0, -2]), np.array([-3, 2])]

        for i, state in enumerate(initial_states, start=1):
            self.agents[i] = Agent(id=i, initial_state=state)
        


        # Barriers
        self.barriers = []
        # self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        
        # Velocity Command Message
        self.max_velocity = 5
        self.vel_cmd_msg = Twist()

        self.task = task_msg()


        # Setup subscribers
        rospy.Subscriber("/tasks", task_msg, self.task_callback)
        rospy.Subscriber("/qualisys/"+self.agent_name+"/pose", PoseStamped, self.pose_callback)
        for id in range(1, self.total_agents+1):
            # rospy.Subscriber(f"/nexus{id}/agent_pose", PoseStamped, self.other_agent_pose_callback)

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

        # Wait until all the task messages have been received
        while len(self.barriers) < 3:
            rospy.sleep(1)


        # Initialize the optimization problem
        # rospy.loginfo(f"barriers {self.agent_id}: {self.barriers}")
        self.relevant_barriers = [barrier for barrier in self.barriers if self.agent_id in barrier.contributing_agents]
        self.solver = self.get_qpsolver_and_parameter_structure()
        self.control_loop()



    def get_qpsolver_and_parameter_structure(self):

        # Create the parameter structure for the optimization problem --- 'p' ---
        parameter_list = []
        parameter_list += [ca_tools.entry(f"state_{self.agent_id}", shape=2)]
        parameter_list += [ca_tools.entry(f"state_{id}", shape=2) for id in self.agents.keys() if id != self.agent_id] # look at this agent
        parameter_list += [ca_tools.entry("time", shape=1)]
        self.parameters = ca_tools.struct_symMX(parameter_list)

        # Create the constraints for the optimization problem --- 'g' ---
        barrier_constraints = self.generate_barrier_constraints(self.relevant_barriers)
        slack_constraints = - ca.vertcat(*list(self.slack_variables.values()))
        constraints = ca.vertcat(barrier_constraints, slack_constraints)

        # Create the decision variables for the optimization problem --- 'x' ---
        slack_vector = ca.vertcat(*list(self.slack_variables.values()))
        opt_vector = ca.vertcat(self.input_vector, slack_vector)

        # Create the object function for the optimization problem --- 'f' ---
        cost = self.input_vector.T @ self.input_vector
        for id,slack in self.slack_variables.items():
            if id == self.agent_id:
                cost += 100* slack**2  
            else :
                cost += 10* slack**2

        # Create the optimization solver
        qp = {'x': opt_vector, 'f': cost, 'g': constraints, 'p': self.parameters}
        solver = ca.qpsol('sol', 'qpoases', qp, {'printLevel': 'none'})

        return solver

        

    def generate_barrier_constraints(self, barrier_list:List[BarrierFunction]) -> ca.MX:
        constraints = []
        for barrier in barrier_list:

            # Check if the barrier has more than one contributing agent
            if len(barrier.contributing_agents) > 1:
                if barrier.contributing_agents[0] == self.agent_id: # can maybe use the list as priority. so the first element is the leader agent
                    neighbour_id = barrier.contributing_agents[1]
                else:
                    neighbour_id = barrier.contributing_agents[0]
            else : 
                neighbour_id = self.agent_id

            # Create the named inputs for the barrier function
            named_inputs = {"state_"+str(id):self.parameters["state_"+str(id)] for id in barrier.contributing_agents}
            named_inputs["time"] = self.parameters["time"]

            # Get the necessary functions from the barrier
            nabla_xi_fun = barrier.gradient_function_wrt_state_of_agent(self.agent_id)
            partial_time_derivative_fun = barrier.partial_time_derivative
            barrier_fun = barrier.function

            # Calculate the symbolic expressions for the barrier constraint
            nabla_xi = nabla_xi_fun.call(named_inputs)["value"]                                        
            dbdt = partial_time_derivative_fun.call(named_inputs)["value"]                             
            alpha_b = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"])       

            # Create load sharing for different constraints
            if neighbour_id == self.agent_id:
                slack = ca.MX.sym('slack', 1)
                self.slack_variables[self.agent_id] = slack
                load_sharing = 1
            else:
                slack = ca.MX.sym('slack', 1)
                self.slack_variables[neighbour_id] = slack
                load_sharing = 0.1
            
            barrier_constraint = -1* ( ca.dot(nabla_xi.T, self.input_vector) + load_sharing * (dbdt + alpha_b + slack))
            constraints.append(barrier_constraint)
            self.nabla_funs.append(nabla_xi_fun)
            self.nabla_inputs.append(named_inputs)

            return ca.vertcat(*constraints)
  


    def control_loop(self):
        r = rospy.Rate(50)
        
        while not rospy.is_shutdown():

            inputs_list = {}
            # Fill the structure with the current values
            current_parameters = self.parameters(0)
            current_parameters["time"] = ca.vertcat(self.last_received_pose.to_sec())
            current_parameters[f'state_{self.agent_id}'] = ca.vertcat(self.agent_pose.pose.position.x, self.agent_pose.pose.position.y)
            for id in self.neighbour_agents.keys():
                current_parameters[f'state_{id}'] = ca.vertcat(self.neighbour_agents[id].pose.position.x, self.neighbour_agents[id].pose.position.y)

            # Calculate the gradient values
            self.nabla_val = []
            for i, nabla_fun in enumerate(self.nabla_funs):
                inputs_list = {key: current_parameters[key] for key in self.nabla_inputs[i].keys()}
                nabla_val = nabla_fun.call(inputs_list)["value"]
                self.nabla_val.append(ca.norm_2(nabla_val))
            # rospy.loginfo(f"gradient{self.agent_id}: {self.nabla_val}")

            # Solve the optimization problem and 
            if any(ca.norm_2(val) < 1e-10 for val in self.nabla_val):
                optimal_input = ca.MX([0, 0])
            else:
                sol = self.solver(p = current_parameters, ubg = 0) 
                optimal_input  = sol['x']
            # rospy.loginfo(f"Optimal_input {self.agent_id}: {sol['x']}")

            # Publish the velocity command 
            # optimal_input = ca.fmin(optimal_input, self.max_velocity)
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
        # state = np.array([msg.pose.position.x, msg.pose.position.y])
        # self.agents[agent_id] = Agent(id=agent_id, initial_state=state)

    def task_callback(self, msg):
        # Store the task message
        self.task = msg

        # rospy.loginfo(f"{self.agent_id} received task message: {self.task}")

        # Create the predicate based on the type of the task
        if self.task.type == "go_to_goal_predicate_2d":
            predicate = go_to_goal_predicate_2d(goal=np.array(self.task.center), epsilon=self.task.epsilon, agent=self.agents[self.task.involved_agents[0]])
        elif self.task.type == "formation_predicate":
            predicate = formation_predicate(epsilon=self.task.epsilon, agent_i=self.agents[self.task.involved_agents[0]], agent_j=self.agents[self.task.involved_agents[1]], relative_pos=np.array(self.task.center))
        # Add more predicates here

        # Create the temporal operator
        temporal_operator = AlwaysOperator(time_interval=TimeInterval(a=self.task.interval[0], b=self.task.interval[1]))

        # Create the task
        task = StlTask(predicate=predicate, temporal_operator=temporal_operator)

        # Add the task to the barriers and the edge
        barrier = create_barrier_from_task(task=task, initial_conditions=[self.agents[i] for i in self.task.involved_agents], alpha_function=self.alpha_fun)
        self.barriers.append(barrier)
        
          


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