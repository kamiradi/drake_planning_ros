#! /usr/bin/env python3

import sys
import numpy as np
import optas

# ros
import rospy
import actionlib
import tempfile

# messages
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
# ROS messages types for the real robot
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
# ROS messages for command configuration action
from iiwa_optas.msg import TriggerCmdAction, TriggerCmdFeedback, TriggerCmdResult

from pydrake.all import (
        DiagramBuilder, AddMultibodyPlantSceneGraph, InverseKinematics,
        RotationMatrix, Solve, RigidTransform, Quaternion, Parser,
        StartMeshcat, MeshcatVisualizer, MeshcatVisualizerParams,
        Simulator, BodyIndex, JointIndex, JointActuatorIndex,
        DifferentialInverseKinematicsParameters, JacobianWrtVariable,
        DifferentialInverseKinematicsStatus,
        DoDifferentialInverseKinematics, DifferentialInverseKinematicsResult)

# For mux controller name
from std_msgs.msg import String
# service for selecting the controller
from topic_tools.srv import MuxSelect

def dict2np(dict: dict) -> np.ndarray:
    return np.array(list(dict.values()))

class CmdTwistActionServer(object):
    """docstring for CmdTwistActionServer."""

    def __init__(self, name):
        # initialization message
        self.name = name
        rospy.loginfo(f'{self.name}: Initializing class')
        # cmd twist action name
        self.cmd_twist_action_server_name = rospy.get_param('~cmd_twist_action_server_name', 'cmd_twist')
        # safety gains on joint position and velocity limits
        self._K_safety_lim_q = rospy.get_param('~K_safety_lim_q', 1.0)
        self._K_safety_lim_q_dot = rospy.get_param('~K_safety_lim_q', 1.0)
        # end-effector frame
        self._link_ee = rospy.get_param('~link_ee', 'link_ee')
        self._link_ref_pos = rospy.get_param('~link_ref_pos', 'world')
        self._link_ref_ori = rospy.get_param('~link_ref_ori', 'world')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        self.dt = 1./self._freq
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        self.joint_feedback = rospy.get_param('~joint_feedback', False)
        # boundaries and gains
        self.ee_min = dict2np(rospy.get_param('~ee_min', {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}))
        self.ee_max = dict2np(rospy.get_param('~ee_max', {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}))
        self._twist_max = dict2np(rospy.get_param('~twist_max', {'x': 0.5, 'y': 0.5, 'z': 0.5, 'roll': 0.1, 'pitch': 0.1, 'yaw': 0.1}))
        self._W = dict2np(rospy.get_param('~W', {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}))
        self._K = rospy.get_param('~K', {'x': 1.0, 'y': 1.0, 'z': 1.0, 'roll': 1.0, 'pitch': 1.0, 'yaw': 1.0})
        #################################################################################
        # initialize variables
        self.twist_target = Twist()
        #################################################################################
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self.name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')

        ### optas
        # set up robot
        robot = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[1]
        )
        self.robot_name = robot.get_name()
        self.ndof = robot.ndof
        # get robot limits
        q_min = self._K_safety_lim_q * robot.dlim[0][0]
        q_max = self._K_safety_lim_q * robot.dlim[0][1]
        dq_min = self._K_safety_lim_q_dot * robot.dlim[1][0]
        dq_max = self._K_safety_lim_q_dot * robot.dlim[1][1]
        # nominal robot configuration
        self.dq_nom = optas.DM.zeros(self.ndof)
        # set up optimization builder
        builder = optas.OptimizationBuilder(T=1, robots=[robot], derivs_align=True)
        # get robot joint variables
        dq = builder.get_model_states(self.robot_name, time_deriv=1)
        # set parameters
        q = builder.add_parameter('q', self.ndof)
        dx = builder.add_decision_variables('dx', 6)
        dx_target = builder.add_parameter('dx_target', 6)
        W_x = builder.add_parameter('W_x', 6)
        ee_min = builder.add_parameter('ee_min', 6)
        ee_max = builder.add_parameter('ee_max', 6)
        delta_twist_max = builder.add_parameter('delta_twist_max', 6)
        # kinematics
        f_pos = robot.get_link_position_function(link=self._link_ee, base_link=self._link_ref_pos)
        J_pos = robot.get_link_linear_jacobian_function(link=self._link_ee, base_link=self._link_ref_pos)
        f_rpy = robot.get_link_rpy_function(link=self._link_ee, base_link=self._link_ref_ori)
        J_rpy = robot.get_link_angular_analytical_jacobian_function(link=self._link_ee, base_link=self._link_ref_ori)
        # cost term
        builder.add_cost_term('cost_q', optas.sumsqr(dq))
        builder.add_cost_term('cost_pos', optas.sumsqr(W_x * (dx_target-dx)))
        # forward differential kinematics
        builder.add_equality_constraint('FDK_pos', (J_pos(q))@dq, dx[:3])
        builder.add_equality_constraint('FDK_ori', (J_rpy(q))@dq, dx[3:])
        # add joint position limits
        builder.add_bound_inequality_constraint('joint_pos_lim', q_min, q+dq, q_max)
        # add joint velocity limitis
        builder.add_bound_inequality_constraint('joint_vel_lim', dq_min, dq, dq_max)
        # add end-effector yaw-pitch-yaw limits
        builder.add_bound_inequality_constraint('ori_lim', optas.deg2rad(ee_min[3:]), f_rpy(q) + J_rpy(q)@dq, optas.deg2rad(ee_max[3:]))
        # add workspace limits
        builder.add_bound_inequality_constraint('pos_lim', ee_min[:3], f_pos(q) + J_pos(q)@dq, ee_max[:3])
        # add end-effector twist bounds
        builder.add_bound_inequality_constraint('twist_bounds', -delta_twist_max, dx, delta_twist_max)
        # setup solver
        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup(
            solver_name='qpoases',
            solver_options={'error_on_fail': False}
        )
        # self.solver = optas.CVXOPTSolver(optimization).setup()
        #################################################################################
        # initialize variables
        self.q_read = np.zeros(self.ndof)
        self.q_cmd = np.zeros(self.ndof)
        self.dq_read = np.zeros(self.ndof)
        self.dq_cmd = np.zeros(self.ndof)
        #################################################################################

        # drake setup
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

        urdf_obj_data = self._robot_description.replace('.stl', '_obj.obj')

        with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.urdf',
                delete=False) as temp_file:
            urdf_file_path = temp_file.name
            temp_file.write(urdf_obj_data)

        parser = Parser(
                self.plant,
                self.scene_graph)
        package_map = parser.package_map()
        # HACK: hard coded package map path
        package_map.Add('iiwa_peg', "/root/workspace/src/mp/iiwa_peg")
        package_map.Add(
                'iiwa_description',
                "/root/workspace/src/robot/iiwa_description")
        package_map.Add(
                'realsense2_description',
                '/opt/ros/noetic/share/realsense2_description')

        robot_instance = parser.AddModels(urdf_file_path)

        self.plant.Finalize()

        # robot variables
        # TODO: self.robot_name
        self.robot_name = "iiwa"
        # TODO: self.ndofS
        self.ndof = self.plant.num_actuators()
        self.gripper_frame = self.plant.GetFrameByName(self.link_ee)

        diagram = builder.Build()
        self.context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self.read_joint_states_callback
        )
        # declare joystick subscriver
        self._joy_sub = rospy.Subscriber(
            '/twist_input',
            Twist,
            self.read_twist_callback
        )
        # declare joint publisher
        self.joint_pub = rospy.Publisher(
            self._pub_cmd_topic_name,
            Float64MultiArray,
            queue_size=10
        )
        # declare twist publisher
        self.twist_pub = rospy.Publisher(
            name='/twist_output',
            data_class=Twist,
            queue_size=10
        )
        # declare compu time publisher
        self.proc_solver_pub = rospy.Publisher(
            name='/proc_time',
            data_class=Float32,
            queue_size=10
        )
        # set mux controller selection as wrong by default
        self._correct_mux_selection = False
        # declare mux service
        self._srv_mux_sel = rospy.ServiceProxy(rospy.get_namespace() + '/mux_joint_position/select', MuxSelect)
        # declare subscriber for selected controller
        self._sub_selected_controller = rospy.Subscriber(
            '/mux_selected',
            String,
            self.read_mux_selection
        )
        # initialize action messages
        self._feedback = TriggerCmdFeedback()
        self._result = TriggerCmdResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            self.cmd_twist_action_server_name, 
            TriggerCmdAction, 
            execute_cb=None,
            auto_start=False
        )
        # register the preempt callback
        self._action_server.register_goal_callback(self.goal_callback)
        self._action_server.register_preempt_callback(self.preempt_callback)
        # start action server
        self._action_server.start()
        #################################################################################

    def goal_callback(self):
        # activate publishing command
        self._srv_mux_sel(self._pub_cmd_topic_name)
        # accept the new goal request
        acceped_goal = self._action_server.accept_new_goal()
        self.target_pos_z = acceped_goal.pos_z
        # read current robot joint positions for memory
        self.q_cmd = self.q_read
        ### optas
        ### ---------------------------------------------------------
        # initialize the joint message
        self.joint_msg = Float64MultiArray()
        self.joint_msg.layout = MultiArrayLayout()
        self.joint_msg.layout.data_offset = 0
        self.joint_msg.layout.dim.append(MultiArrayDimension())
        self.joint_msg.layout.dim[0].label = "columns"
        self.joint_msg.layout.dim[0].size = self.ndof
        # initialize the twist message
        self.twist_msg = Twist()
        # initialize the float message
        self.float_msg = Float32()

        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_callback)

    def timer_callback(self, event):
        """ Publish the robot configuration """

        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn(f'{self.name}: The action server is NOT active!')
            self._result.trigger_off = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._correct_mux_selection):
            # current config
            if self.joint_feedback:
                q_curr = self.q_read
            else:
                q_curr = self.q_cmd
            # compute forward kinematics
            dx_target = [
                self._K['x'] * self.dt * self.twist_target.linear.x,
                self._K['y'] * self.dt * self.twist_target.linear.y,
                self._K['z'] * self.dt * self.twist_target.linear.z,
                self._K['roll'] * self.dt * self.twist_target.angular.x,
                self._K['pitch'] * self.dt * self.twist_target.angular.y,
                self._K['yaw'] * self.dt * self.twist_target.angular.z
            ]

            # setup DoDifferentialIK here
            # NOTE: add drake diffik here
            diff_ik_params = DifferentialInverseKinematicsParameters(
                self.ndofs,
                self.ndofs)
            diff_ik_params.set_nominal_joint_position(q_curr)
            diff_ik_params.set_end_effector_translational_velocity_limits(
                -1*self.dt*self._twist_max[0:3],
                self.dt*self._twist_max[0:3],
                )
            diff_ik_params.set_end_effector_angular_speed_limit(0)

            dq_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
            dq_min = -1*dq_limits
            dq_max = dq_limits
            diff_ik_params.set_joint_velocity_limits(
                (dq_min, dq_max))


            # NOTE: can be deleted from here
            self.solver.reset_parameters({
                'q': q_curr,
                'dx_target': dx_target,
                'ee_min': self.ee_min,
                'ee_max': self.ee_max,
                'delta_twist_max': self.dt * self._twist_max,
                'W_x': self._W,
            })
            self.solver.reset_initial_seed({f'{self.robot_name}/dq': self.dq_nom})
            # solve problem
            solution = self.solver.solve()
            stats = self.solver.stats()
            t_proc_solver = stats['t_proc_solver']
            if self.solver.did_solve():
                dq = np.asarray(solution[f'{self.robot_name}/dq']).T[0]
                dx = np.asarray(solution['dx']).T[0] / self.dt
            else:
                rospy.logwarn(f'{self.name}: The QP fail to find a solution!')
                dq = np.zeros(self.ndof)
                dx = np.zeros(6)

            # NOTE: don't change from here
            # integrate solution
            self.q_cmd = q_curr + dq
            # update message
            self.joint_msg.data = self.q_cmd
            self.twist_msg.linear.x = dx[0]
            self.twist_msg.linear.y = dx[1]
            self.twist_msg.linear.z = dx[2]
            self.twist_msg.angular.x = dx[3]
            self.twist_msg.angular.y = dx[4]
            self.twist_msg.angular.z = dx[5]
            self.float_msg.data = t_proc_solver
            # publish message
            self.joint_pub.publish(self.joint_msg)
            self.twist_pub.publish(self.twist_msg)
            self.proc_solver_pub.publish(self.float_msg)
            # compute progress
            self._feedback.is_active = True
            # publish feedback
            self._action_server.publish_feedback(self._feedback)
        else:
            # shutdown this timer
            self._timer.shutdown()
            rospy.logwarn(f'{self.name}: Request aborted. The controller selection changed!')
            self._result.trigger_off = False
            self._action_server.set_aborted(self._result)
            return

    def read_twist_callback(self, msg):
        self.twist_target = msg

    def read_joint_states_callback(self, msg):
        self.q_read = np.asarray(list(msg.position))

    def read_mux_selection(self, msg):
        rospy.loginfo(f'Mux selection{msg.data}')
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_callback(self):
        self._timer.shutdown()
        rospy.loginfo(f'{self.name}: Client preempted this action.')
        self._result.trigger_off = True
        # set the action state to preempted
        self._action_server.set_preempted(self._result)


class DiffIKActionServer(object):
    """docstring for CmdTwistActionServer."""

    def __init__(self, name):
        # initialization message
        self.name = name
        rospy.loginfo(f'{self.name}: Initializing class')
        # cmd twist action name
        self.cmd_twist_action_server_name = rospy.get_param('~cmd_twist_action_server_name', 'cmd_twist')
        # safety gains on joint position and velocity limits
        self._K_safety_lim_q = rospy.get_param('~K_safety_lim_q', 1.0)
        self._K_safety_lim_q_dot = rospy.get_param('~K_safety_lim_q', 1.0)
        # end-effector frame
        self._link_ee = rospy.get_param('~link_ee', 'link_ee')
        self._link_ref_pos = rospy.get_param('~link_ref_pos', 'world')
        self._link_ref_ori = rospy.get_param('~link_ref_ori', 'world')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        self.dt = 1./self._freq
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        self.joint_feedback = rospy.get_param('~joint_feedback', False)
        # boundaries and gains
        self.ee_min = dict2np(rospy.get_param(
            '~ee_min',
            {'x': 0.0,
             'y': 0.0,
             'z': 0.0,
             'roll': 0.0,
             'pitch': 0.0,
             'yaw': 0.0}))
        self.ee_max = dict2np(rospy.get_param(
            '~ee_max', {'x': 0.0,
                        'y': 0.0,
                        'z': 0.0,
                        'roll': 0.0,
                        'pitch': 0.0,
                        'yaw': 0.0}))
        self._twist_max = dict2np(rospy.get_param(
            '~twist_max',
            {'x': 0.5,
             'y': 0.5,
             'z': 0.5,
             'roll': 0.1,
             'pitch': 0.1,
             'yaw': 0.1}))
        self._W = dict2np(rospy.get_param(
            '~W',
            {'x': 0,
             'y': 0,
             'z': 0,
             'roll': 0,
             'pitch': 0,
             'yaw': 0}))
        self._K = rospy.get_param(
            '~K',
            {'x': 1.0,
             'y': 1.0,
             'z': 1.0,
             'roll': 1.0,
             'pitch': 1.0,
             'yaw': 1.0})

        # initialize variables
        self.twist_target = Twist()
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self.name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')


        # drake setup
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

        urdf_obj_data = self._robot_description.replace('.stl', '_obj.obj')

        with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.urdf',
                delete=False) as temp_file:
            urdf_file_path = temp_file.name
            temp_file.write(urdf_obj_data)

        parser = Parser(
                self.plant,
                self.scene_graph)
        package_map = parser.package_map()
        # HACK: hard coded package map path
        package_map.Add('iiwa_peg', "/root/workspace/src/mp/iiwa_peg")
        package_map.Add(
                'iiwa_description',
                "/root/workspace/src/robot/iiwa_description")
        package_map.Add(
                'realsense2_description',
                '/opt/ros/noetic/share/realsense2_description')

        robot_instance = parser.AddModels(urdf_file_path)

        self.plant.Finalize()

        # robot variables
        # TODO: self.robot_name
        self.robot_name = "iiwa"
        # TODO: self.ndofS
        self.ndof = self.plant.num_actuators()
        self.gripper_frame = self.plant.GetFrameByName(self._link_ee)

        diagram = builder.Build()
        self.context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        # initialize variables
        self.q_read = np.zeros(self.ndof)
        self.q_cmd = np.zeros(self.ndof)
        self.dq_read = np.zeros(self.ndof)
        self.dq_cmd = np.zeros(self.ndof)

        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self.read_joint_states_callback
        )
        # declare joystick subscriver
        self._joy_sub = rospy.Subscriber(
            '/twist_input',
            Twist,
            self.read_twist_callback
        )
        # declare joint publisher
        self.joint_pub = rospy.Publisher(
            self._pub_cmd_topic_name,
            Float64MultiArray,
            queue_size=10
        )
        # declare twist publisher
        self.twist_pub = rospy.Publisher(
            name='/twist_output',
            data_class=Twist,
            queue_size=10
        )
        # declare compu time publisher
        self.proc_solver_pub = rospy.Publisher(
            name='/proc_time',
            data_class=Float32,
            queue_size=10
        )
        # set mux controller selection as wrong by default
        self._correct_mux_selection = False
        # declare mux service
        self._srv_mux_sel = rospy.ServiceProxy(
                rospy.get_namespace() + '/mux_joint_position/select', MuxSelect)
        # declare subscriber for selected controller
        self._sub_selected_controller = rospy.Subscriber(
            '/mux_selected',
            String,
            self.read_mux_selection
        )
        # initialize action messages
        self._feedback = TriggerCmdFeedback()
        self._result = TriggerCmdResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            self.cmd_twist_action_server_name,
            TriggerCmdAction,
            execute_cb=None,
            auto_start=False
        )
        # register the preempt callback
        self._action_server.register_goal_callback(self.goal_callback)
        self._action_server.register_preempt_callback(self.preempt_callback)
        # start action server
        self._action_server.start()
        #################################################################################

    def goal_callback(self):
        # activate publishing command
        self._srv_mux_sel(self._pub_cmd_topic_name)
        # accept the new goal request
        acceped_goal = self._action_server.accept_new_goal()
        self.target_pos_z = acceped_goal.pos_z
        # read current robot joint positions for memory
        self.q_cmd = self.q_read
        # initialize the joint message
        self.joint_msg = Float64MultiArray()
        self.joint_msg.layout = MultiArrayLayout()
        self.joint_msg.layout.data_offset = 0
        self.joint_msg.layout.dim.append(MultiArrayDimension())
        self.joint_msg.layout.dim[0].label = "columns"
        self.joint_msg.layout.dim[0].size = self.ndof
        # initialize the twist message
        self.twist_msg = Twist()
        # initialize the float message
        self.float_msg = Float32()

        # create timer
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_callback)

    def timer_callback(self, event):
        """ Publish the robot configuration """

        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn(f'{self.name}: The action server is NOT active!')
            self._result.trigger_off = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._correct_mux_selection):
            # current config
            if self.joint_feedback:
                q_curr = self.q_read
                dq_curr = self.dq_read
            else:
                q_curr = self.q_cmd
                dq_curr = self.dq_cmd
            # compute forward kinematics
            dx_target = [
                self._K['x'] * self.dt * self.twist_target.linear.x,
                self._K['y'] * self.dt * self.twist_target.linear.y,
                self._K['z'] * self.dt * self.twist_target.linear.z,
                self._K['roll'] * self.dt * self.twist_target.angular.x,
                self._K['pitch'] * self.dt * self.twist_target.angular.y,
                self._K['yaw'] * self.dt * self.twist_target.angular.z
            ]

            # setup DoDifferentialIK here
            # NOTE: add drake diffik here
            diff_ik_params = DifferentialInverseKinematicsParameters(
                self.ndof,
                self.ndof)
            diff_ik_params.set_nominal_joint_position(q_curr)
            velocity_flag = np.asarray([False, False, False, True, True, True])
            diff_ik_params.set_end_effector_velocity_flag(velocity_flag)
            diff_ik_params.set_end_effector_translational_velocity_limits(
                -1*self.dt*self._twist_max[0:3],
                self.dt*self._twist_max[0:3],
                )
            diff_ik_params.set_end_effector_angular_speed_limit(0)

            dq_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
            dq_min = -1*dq_limits
            dq_max = dq_limits
            diff_ik_params.set_joint_velocity_limits(
                (dq_min, dq_max))

            # get the jacobian
            self.plant.SetPositions(self.plant_context, q_curr)
            self.plant.SetVelocities(self.plant_context, dq_curr)
            J_G = self.plant.CalcJacobianSpatialVelocity(
                self.plant_context,
                JacobianWrtVariable.kQDot,
                self.gripper_frame,
                [0, 0, 0],
                self.plant.world_frame(),
                self.plant.world_frame()
                )
            result = DoDifferentialInverseKinematics(
                q_curr,
                dq_curr,
                dx_target,
                J_G,
                diff_ik_params
                )


            if result.status == DifferentialInverseKinematicsStatus.kSolutionFound:
                dq = result.joint_velocities
            else:
                rospy.logwarn(f'{self.name}: The QP fail to find a solution!')
                dq = np.zeros(self.ndof)
                # dx = np.zeros(6)

            # NOTE: don't change from here
            # integrate solution
            self.q_cmd = q_curr + dq
            # update message
            self.joint_msg.data = self.q_cmd
            # publish message
            self.joint_pub.publish(self.joint_msg)
            # self.twist_pub.publish(self.twist_msg)
            # self.proc_solver_pub.publish(self.float_msg)
            # compute progress
            self._feedback.is_active = True
            # publish feedback
            self._action_server.publish_feedback(self._feedback)
        else:
            # shutdown this timer
            self._timer.shutdown()
            rospy.logwarn(f'{self.name}: Request aborted. The controller selection changed!')
            self._result.trigger_off = False
            self._action_server.set_aborted(self._result)
            return

    def read_twist_callback(self, msg):
        self.twist_target = msg

    def read_joint_states_callback(self, msg):
        self.q_read = np.asarray(list(msg.position))
        self.dq_read = np.asarray(list(msg.velocity))

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_callback(self):
        self._timer.shutdown()
        rospy.loginfo(f'{self.name}: Client preempted this action.')
        self._result.trigger_off = True
        # set the action state to preempted
        self._action_server.set_preempted(self._result)


if __name__ == "__main__":
    # Initialize node
    rospy.init_node('teleop_2d_server', anonymous=True)
    # Initialize node class
    teleop_2d_server = DiffIKActionServer(rospy.get_name())
    # executing node
    rospy.spin()
