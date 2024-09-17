#! /usr/bin/env python3

# Copyright (C) 2022 Statistical Machine Learning and Motor Control Group (SLMC)
# Authors: Joao Moura (maintainer)
# email: joao.moura@ed.ac.uk

# This file is part of iiwa_optas package.

# iiwa_optas is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# iiwa_optas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
import numpy as np
from scipy.spatial.transform import Rotation as R
import optas

# drake
from pydrake.all import (
        DiagramBuilder, AddMultibodyPlantSceneGraph, InverseKinematics,
        RotationMatrix, Solve)

# ros
import rospy
import actionlib

# messages
from sensor_msgs.msg import JointState
# ROS messages types for the real robot
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
# ROS messages for command configuration action
from iiwa_optas.msg import CmdPoseAction, CmdPoseFeedback, CmdPoseResult
from drake_planning_ros.msg import IKPoseAction, IKPoseFeedback, IKPoseResult
from drake_planning_ros.utils import *

# For mux controller name
from std_msgs.msg import String
# service for selecting the controller
from topic_tools.srv import MuxSelect
np.set_printoptions(precision=2)


def dict2np(dict: dict) -> np.ndarray:
    return np.array(list(dict.values()))


class CmdPoseActionServer(object):
    """docstring for CmdPoseActionServer."""

    def __init__(self, name):
        # initialization message
        self.name = name
        rospy.loginfo(f'{self.name}: Initializing class')

        # get parameters:
        # --------------------------------------
        # workspace limit boundaries
        self.pos_min = dict2np(rospy.get_param('~pos_min', {'x': -0.7, 'y': -0.5, 'z': 0.02}))
        self.pos_max = dict2np(rospy.get_param('~pos_max', {'x': -0.5, 'y': 0.2, 'z': 0.3}))
        # end-effector frame
        self.link_ee = rospy.get_param('~link_ee', 'link_ee')
        self.link_ref = rospy.get_param('~link_ref', 'world')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param('~cmd_topic_name', '/command')
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
        else:
            rospy.logerr("%s: Param %s is unavailable!" % (self.name, param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')

        # optas
        # ---------------------------------------------------------
        # set up robot
        robot = optas.RobotModel(
            urdf_string=self._robot_description,
            time_derivs=[0]
        )
        self.robot_name = robot.get_name()
        self.ndof = robot.ndof
        # set up optimization builder
        builder = optas.OptimizationBuilder(T=1, robots=[robot])
        # get robot state variables
        q_var = builder.get_model_states(self.robot_name)
        # nominal robot configuration
        q_nom = builder.add_parameter('q_nom', self.ndof)
        # get end-effector pose as parameters
        pos = builder.add_parameter('pos', 3)
        ori = builder.add_parameter('ori', 4)
        # set variable boudaries
        builder.enforce_model_limits(self.robot_name)
        # equality constraint on position
        pos_fnc = robot.get_global_link_position_function(link=self.link_ee)
        builder.add_equality_constraint('final_pos', pos_fnc(q_var), rhs=pos)
        # equality constraint on orientation
        ori_fnc = robot.get_global_link_quaternion_function(link=self.link_ee)
        ori_fnc= robot.get_link_quaternion_function(link=self.link_ee, base_link=self.link_ref)
        builder.add_equality_constraint('final_ori', ori_fnc(q_var), rhs=ori)
        # optimization cost: close to nominal config
        builder.add_cost_term('nom_config', optas.sumsqr(q_var-q_nom))
        # setup solver
        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization=optimization).setup('ipopt')
        ### ---------------------------------------------------------
        # initialize variables
        self.q_curr = np.zeros(self.ndof)
        # declare joint subscriber
        self._joint_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self.read_joint_states_callback
        )
        # declare joint publisher
        self._joint_pub = rospy.Publisher(
            self._pub_cmd_topic_name,
            Float64MultiArray,
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
        self._feedback = CmdPoseFeedback()
        self._result = CmdPoseResult()
        # declare action server
        self._action_server = actionlib.SimpleActionServer(
            'cmd_pose', 
            CmdPoseAction, 
            execute_cb=None,
            auto_start=False
        )
        # register the preempt callback
        self._action_server.register_goal_callback(self.goal_callback)
        self._action_server.register_preempt_callback(self.preempt_callback)
        # start action server
        self._action_server.start()

    def goal_callback(self):
        # activate publishing command
        self._srv_mux_sel(self._pub_cmd_topic_name)
        # accept the new goal request
        acceped_goal = self._action_server.accept_new_goal()
        # desired end-effector position
        pos_T = np.asarray([
                acceped_goal.pose.position.x,
                acceped_goal.pose.position.y,
                acceped_goal.pose.position.z
        ])
        ori_T = np.asarray([
                acceped_goal.pose.orientation.x,
                acceped_goal.pose.orientation.y,
                acceped_goal.pose.orientation.z,
                acceped_goal.pose.orientation.w
        ])
        # check boundaries of the position
        if (pos_T > self.pos_max).any() or (pos_T < self.pos_min).any():
            rospy.logwarn("%s: Request aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self.name, pos_T[0], pos_T[1], pos_T[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo('%s: Request to send robot to position (%.2f, %.2f, %.2f) and orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds.' % (
                self.name, 
                pos_T[0], pos_T[1], pos_T[2],
                ori_T[0], ori_T[1], ori_T[2], ori_T[3],
                acceped_goal.duration
            )
        )
        # read current robot joint positions
        q0 = self.q_curr
        # get nominal joint positions
        q_nom = np.asarray(list(acceped_goal.nom_joint_pos))
        if q_nom.size == 0:
            q_nom = q0
        if not (q_nom.size == self.ndof):
            rospy.logerr(f"{self.name}: Nominal config {q_nom} should containt {self.ndof} positions.")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        ### optas
        ### ---------------------------------------------------------
        # set initial seed
        self.solver.reset_initial_seed({f'{self.robot_name}/q': q0})
        self.solver.reset_parameters({
            'pos': pos_T,
            'ori': ori_T,
            'q_nom': q_nom,
        })
        # solve problem
        solution = self.solver.solve()
        if self.solver.did_solve():
            qT_array = solution[f'{self.robot_name}/q']
            ### ---------------------------------------------------------
            T = acceped_goal.duration
            qT = np.asarray(qT_array).T[0]
            rospy.loginfo(f"{self.name}: Sending robot to config {qT}.")
            # helper variables
            self._steps = int(T * self._freq)
            self._idx = 0
            Dq = qT - q0
            # interpolate between current and target configuration 
            self._q = lambda t: q0 + (10.*((t/T)**3) - 15.*((t/T)**4) + 6.*((t/T)**5))*Dq # 5th order
            self._t = np.linspace(0., T, self._steps + 1)
            # initialize the message
            self._msg = Float64MultiArray()
            self._msg.layout = MultiArrayLayout()
            self._msg.layout.data_offset = 0
            self._msg.layout.dim.append(MultiArrayDimension())
            self._msg.layout.dim[0].label = "columns"
            self._msg.layout.dim[0].size = self.ndof
            # create timer
            dur = rospy.Duration(1.0/self._freq)
            self._timer = rospy.Timer(dur, self.timer_callback)
        else:
            rospy.logerr(f"{self.name}: Failed to find a configuration for the desired pose.")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)


    def timer_callback(self, event):
        """ Publish the robot configuration """

        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn(f'{self.name}: The action server is NOT active!')
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._idx < self._steps):
            if(self._correct_mux_selection):
                # increment idx (in here counts starts with 1)
                self._idx += 1
                # compute next configuration with lambda function
                q_next = self._q(self._t[self._idx])
                # update message
                self._msg.data = q_next
                # publish message
                self._joint_pub.publish(self._msg)
                # compute progress
                self._feedback.progress = (self._idx*100)/self._steps
                # publish feedback
                self._action_server.publish_feedback(self._feedback)
            else:
                # shutdown this timer
                self._timer.shutdown()
                rospy.logwarn(f'{self.name}: Request aborted. The controller selection changed!')
                self._result.reached_goal = False
                self._action_server.set_aborted(self._result)
                return
        else:
            # shutdown this timer
            self._timer.shutdown()
            # set the action state to succeeded
            rospy.loginfo(f'{self.name}: Succeeded')
            self._result.reached_goal = True
            self._action_server.set_succeeded(self._result)
            return

    def read_joint_states_callback(self, msg):
        self.q_curr = np.asarray(list(msg.position))

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_callback(self):
        rospy.loginfo(f'{self.name}: Preempted.')
        # set the action state to preempted
        self._action_server.set_preempted()


class IkPoseActionServer(object):
    """
    Action server that sets up and computes an IK solution to a desired task
    space coordinate. It then constructs a 5th order spline trajectory between
    current configuration and the computed configuration.
    """

    def __init__(self, name):

        # initialization message
        self.name = name
        rospy.loginfo(f'{self.name}: Initializing class')

        # get parameters
        self.pos_min = dict2np(
                rospy.get_param('~pos_min', {
                    'x': -0.7,
                    'y': -0.5,
                    'z': 0.02}))
        self.pos_max = dict2np(
                rospy.get_param('~pos_max', {
                    'x': -0.5,
                    'y': 0.2,
                    'z': 0.3}))
        # end-effector frame
        self.link_ee = rospy.get_param('~link_ee', 'link_ee')
        self.link_ref = rospy.get_param('~link_ref', 'world')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param(
                '~cmd_topic_name',
                '/command')
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
        else:
            rospy.logerr(
                    "%s: Param %s is unavailable!" % (
                        self.name,
                        param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')

        # drake
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

        # TODO: add the relevant urdf here

        self.plant.Finalize()

        diagram = builder.Build()
        self.context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        q0 = self.plant.GetPositions(self.plant_context)

        # setup action server
        self._feedback = IKPoseFeedback()
        self._result = IKPoseResult()

        # declare action server
        self._action_server = actionlib.SimpleActionServer(
                'ik_pose',
                IKPoseAction,
                execute_cb=None,
                auto_start=False)

        # register goal and preempt callback
        self._action_server.register_goal_callback(self.goal_callback)
        self._action_server.register_preempt_callback(self.preempt_callback)
        # start action server
        self._action_server.start()

    def goal_callback(self):

        # accept the new goal request
        accepted_goal = self._action_server.accept_new_goal()
        # desired end-effector position
        pos_T = np.asarray([
                accepted_goal.pose.position.x,
                accepted_goal.pose.position.y,
                accepted_goal.pose.position.z
        ])
        ori_T = np.asarray([
                accepted_goal.pose.orientation.x,
                accepted_goal.pose.orientation.y,
                accepted_goal.pose.orientation.z,
                accepted_goal.pose.orientation.w
        ])
        # check boundaries of the position
        if (pos_T > self.pos_max).any() or (pos_T < self.pos_min).any():
            rospy.logwarn("%s: Request aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self.name, pos_T[0], pos_T[1], pos_T[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo(
                '%s: Request to send robot to position (%.2f, %.2f, %.2f) and orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds.' % (
                self.name,
                pos_T[0], pos_T[1], pos_T[2],
                ori_T[0], ori_T[1], ori_T[2], ori_T[3],
                accepted_goal.duration
            )
        )

        # setup optimization
        q0 = self.q_curr
        q_nom = np.asarray(list(accepted_goal.nom_joint_pos))
        if q_nom.size == 0:
            q_nom = q0
        if not (q_nom.size == self.ndof):
            rospy.logerr(f"{self.name}: Nominal config {q_nom} should containt {self.ndof} positions.")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        goal_pose = from_ros_pose(accepted_goal.pose)
        ik = InverseKinematics(
                self.plant,
                self.plant_context)
        ik.AddPositionConstraint(
            self.gripper_frame,
            [0, 0, 0],
            self.plant.world_frame(),
            goal_pose.translation(),
            goal_pose.translation()
            )
        ik.AddOrientationConstraint(
            self.gripper_frame,
            RotationMatrix(),
            self.plant.world_frame(),
            goal_pose.rotation(),
            0.0
            )

        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)
        result = Solve(ik.prog())

        if result.is_success():
            # qT_array = solution[f'{self.robot_name}/q']
            qT_array = result.GetSolution()
            ### ---------------------------------------------------------
            T = acceped_goal.duration
            qT = np.asarray(qT_array).T[0]
            rospy.loginfo(f"{self.name}: Sending robot to config {qT}.")
            # helper variables
            self._steps = int(T * self._freq)
            self._idx = 0
            Dq = qT - q0
            # interpolate between current and target configuration 
            self._q = lambda t: q0 + (10.*((t/T)**3) - 15.*((t/T)**4) + 6.*((t/T)**5))*Dq # 5th order
            self._t = np.linspace(0., T, self._steps + 1)
            # initialize the message
            self._msg = Float64MultiArray()
            self._msg.layout = MultiArrayLayout()
            self._msg.layout.data_offset = 0
            self._msg.layout.dim.append(MultiArrayDimension())
            self._msg.layout.dim[0].label = "columns"
            self._msg.layout.dim[0].size = self.ndof
            # create timer
            dur = rospy.Duration(1.0/self._freq)
            self._timer = rospy.Timer(dur, self.timer_callback)
            pass
        else:
            rospy.logerr(f"{self.name}: Failed to find a configuration for the desired pose.")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
        return

    def timer_callback(self, event):
        """ Publish the robot configuration """

        # make sure that the action is active
        if(not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn(f'{self.name}: The action server is NOT active!')
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        # main execution
        if(self._idx < self._steps):
            if(self._correct_mux_selection):
                # increment idx (in here counts starts with 1)
                self._idx += 1
                # compute next configuration with lambda function
                q_next = self._q(self._t[self._idx])
                # update message
                self._msg.data = q_next
                # publish message
                self._joint_pub.publish(self._msg)
                # compute progress
                self._feedback.progress = (self._idx*100)/self._steps
                # publish feedback
                self._action_server.publish_feedback(self._feedback)
            else:
                # shutdown this timer
                self._timer.shutdown()
                rospy.logwarn(f'{self.name}: Request aborted. The controller selection changed!')
                self._result.reached_goal = False
                self._action_server.set_aborted(self._result)
                return
        else:
            # shutdown this timer
            self._timer.shutdown()
            # set the action state to succeeded
            rospy.loginfo(f'{self.name}: Succeeded')
            self._result.reached_goal = True
            self._action_server.set_succeeded(self._result)
            return

    def read_joint_states_callback(self, msg):
        self.q_curr = np.asarray(list(msg.position))

    def read_mux_selection(self, msg):
        self._correct_mux_selection = (msg.data == self._pub_cmd_topic_name)

    def preempt_callback(self):
        rospy.loginfo(f'{self.name}: Preempted.')
        # set the action state to preempted
        self._action_server.set_preempted()


class IkPoseActionServer(object):
    """
    Action server that sets up and computes an IK solution to a desired task
    space coordinate. It then constructs a 5th order spline trajectory between
    current configuration and the computed configuration.
    """

    def __init__(self, name):

        # initialization message
        self.name = name
        rospy.loginfo(f'{self.name}: Initializing class')

        # get parameters
        self.pos_min = dict2np(
                rospy.get_param('~pos_min', {
                    'x': -0.7,
                    'y': -0.5,
                    'z': 0.02}))
        self.pos_max = dict2np(
                rospy.get_param('~pos_max', {
                    'x': -0.5,
                    'y': 0.2,
                    'z': 0.3}))
        # end-effector frame
        self.link_ee = rospy.get_param('~link_ee', 'link_ee')
        self.link_ref = rospy.get_param('~link_ref', 'world')
        # control frequency
        self._freq = rospy.get_param('~freq', 100)
        # publishing command node name
        self._pub_cmd_topic_name = rospy.get_param(
                '~cmd_topic_name',
                '/command')
        # load robot_description
        param_robot_description = '~/robot_description'
        if rospy.has_param(param_robot_description):
            self._robot_description = rospy.get_param(param_robot_description)
        else:
            rospy.logerr(
                    "%s: Param %s is unavailable!" % (
                        self.name,
                        param_robot_description))
            rospy.signal_shutdown('Incorrect parameter name.')

        # drake
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

        # TODO: add the relevant urdf here

        self.plant.Finalize()

        diagram = builder.Build()
        self.context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        q0 = self.plant.GetPositions(self.plant_context)

        # setup action server
        self._feedback = IKPoseFeedback()
        self._result = IKPoseResult()

        # declare action server
        self._action_server = actionlib.SimpleActionServer(
                'ik_pose',
                IKPoseAction,
                execute_cb=None,
                auto_start=False)

        # register goal and preempt callback
        self._action_server.register_goal_callback(self.goal_callback)
        self._action_server.register_preempt_callback(self.preempt_callback)
        # start action server
        self._action_server.start()

    def goal_callback(self):

        # accept the new goal request
        accepted_goal = self._action_server.accept_new_goal()
        # desired end-effector position
        pos_T = np.asarray([
                accepted_goal.pose.position.x,
                accepted_goal.pose.position.y,
                accepted_goal.pose.position.z
        ])
        ori_T = np.asarray([
                accepted_goal.pose.orientation.x,
                accepted_goal.pose.orientation.y,
                accepted_goal.pose.orientation.z,
                accepted_goal.pose.orientation.w
        ])
        # check boundaries of the position
        if (pos_T > self.pos_max).any() or (pos_T < self.pos_min).any():
            rospy.logwarn("%s: Request aborted. Goal position (%.2f, %.2f, %.2f) is outside of the workspace boundaries. Check parameters for this node." % (self.name, pos_T[0], pos_T[1], pos_T[2]))
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        # print goal request
        rospy.loginfo(
                '%s: Request to send robot to position (%.2f, %.2f, %.2f) and orientation (%.2f, %.2f, %.2f, %.2f) in %.1f seconds.' % (
                self.name,
                pos_T[0], pos_T[1], pos_T[2],
                ori_T[0], ori_T[1], ori_T[2], ori_T[3],
                accepted_goal.duration
            )
        )

        # setup optimization
        q0 = self.q_curr
        q_nom = np.asarray(list(accepted_goal.nom_joint_pos))
        if q_nom.size == 0:
            q_nom = q0
        if not (q_nom.size == self.ndof):
            rospy.logerr(f"{self.name}: Nominal config {q_nom} should containt {self.ndof} positions.")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return

        goal_pose = from_ros_pose(accepted_goal.pose)
        ik = InverseKinematics(
                self.plant,
                self.plant_context)
        ik.AddPositionConstraint(
            self.gripper_frame,
            [0, 0, 0],
            self.plant.world_frame(),
            goal_pose.translation(),
            goal_pose.translation()
            )
        ik.AddOrientationConstraint(
            self.gripper_frame,
            RotationMatrix(),
            self.plant.world_frame(),
            goal_pose.rotation(),
            0.0
            )

        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)
        result = Solve(ik.prog())

        if result.is_success():
            # qT_array = solution[f'{self.robot_name}/q']
            qT_array = result.GetSolution()
            ### ---------------------------------------------------------
            T = acceped_goal.duration
            qT = np.asarray(qT_array).T[0]
            rospy.loginfo(f"{self.name}: Sending robot to config {qT}.")
            # helper variables
            self._steps = int(T * self._freq)
            self._idx = 0
            Dq = qT - q0
            # interpolate between current and target configuration 
            self._q = lambda t: q0 + (10.*((t/T)**3) - 15.*((t/T)**4) + 6.*((t/T)**5))*Dq # 5th order
            self._t = np.linspace(0., T, self._steps + 1)
            # initialize the message
            self._msg = Float64MultiArray()
            self._msg.layout = MultiArrayLayout()
            self._msg.layout.data_offset = 0
            self._msg.layout.dim.append(MultiArrayDimension())
            self._msg.layout.dim[0].label = "columns"
            self._msg.layout.dim[0].size = self.ndof
            # create timer
            dur = rospy.Duration(1.0/self._freq)
            self._timer = rospy.Timer(dur, self.timer_callback)
            pass
        else:
            rospy.logerr(f"{self.name}: Failed to find a configuration for the desired pose.")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
        return



if __name__=='__main__':
    # Initialize node
    rospy.init_node('cmd_pose_server', anonymous=True)
    # Initialize node class
    cmd_pose_server = CmdPoseActionServer(rospy.get_name())
    # executing node
    rospy.spin()
