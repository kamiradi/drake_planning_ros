#!/usr/bin/python
import rospy
import actionlib
import sys
import tf2_ros
import numpy as np
import geometry_msgs.msg as geom_msg
import sensor_msgs.msg as sense_msg
import control_msgs.msg as ctrl_msg
from std_srvs.srv import Empty
from dynamic_reconfigure.client import Client
from switching_system.msg import CmdPoseAction, CmdPoseFeedback, CmdPoseResult
from pydrake.all import (
    RigidTransform, Quaternion, RollPitchYaw, PiecewisePose
)


def _write_pose_msg(X_AB, p, q):
    # p - position message
    # q - quaternion message
    X_AB = RigidTransform(X_AB)
    p.x, p.y, p.z = X_AB.translation()
    q.w, q.x, q.y, q.z = X_AB.rotation().ToQuaternion().wxyz()


def to_ros_transform(X_AB):
    """Converts Drake transform to ROS transform."""
    msg = geom_msg.Transform()
    _write_pose_msg(X_AB, p=msg.translation, q=msg.rotation)
    return msg


def to_ros_pose(X_AB):
    """Converts Drake transform to ROS pose."""
    msg = geom_msg.Pose()
    _write_pose_msg(X_AB, p=msg.position, q=msg.orientation)
    return msg


def _read_pose_msg(p, q):
    # p - position message
    # q - quaternion message
    return RigidTransform(
        Quaternion(wxyz=[q.w, q.x, q.y, q.z]), [p.x, p.y, p.z])


def from_ros_pose(pose):
    """Converts ROS pose to Drake transform."""
    return _read_pose_msg(p=pose.position, q=pose.orientation)


def from_ros_transform(tr):
    """Converts ROS transform to Drake transform."""
    return _read_pose_msg(p=tr.translation, q=tr.rotation)


def MakePegPoseTrajectory(X_G, times):
    sample_times = []
    poses = []
    for name in ["initial", "final"]:
        sample_times.append(times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


class CmdPoseActionServer(object):
    """
    This class implements an action server that moves the robot to a given pose
    by the client
    """

    def __init__(self, name, debug=True):
        # class related variables
        self._name = name
        self._freq = 100
        self._timer = None
        self._idx = 0
        self._steps = 0
        self._velocity = 0.05
        self._plan = None
        self._plan_start_time = None
        self._debug = debug
        self.start_srv = "/ae_ros/start_estimation"
        self.stop_srv = "/ae_ros/stop_estimation"
        self.reset_srv = "/ae_ros/reset_estimation"
        self.use_estimation = False

        # some subscribers and publishers
        self._equilibrium_pose_pub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=1
        )
        self._gripper_cmd_pub = rospy.Publisher(
            "/gripper_controller/gripper_cmd/goal",
            ctrl_msg.GripperCommandActionGoal,
            queue_size=1
        )
        self.compliance_topic = "/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
        self.client = Client(self.compliance_topic)
        rospy.loginfo(
            "Command pose action server paused until estimation services found")

        # start estimation service
        rospy.wait_for_service(self.start_srv)
        try:
            self._start_estimation_srv = rospy.ServiceProxy(
                self.start_srv, Empty)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to find service: %s", e)

        # stop estimation service
        rospy.wait_for_service(self.stop_srv)
        try:
            self._stop_estimation_srv = rospy.ServiceProxy(
                self.stop_srv, Empty)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to find service: %s", e)

        # reset estimation service
        rospy.wait_for_service(self.reset_srv)
        try:
            self._reset_estimation_srv = rospy.ServiceProxy(
                self.reset_srv, Empty)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to find service: %s", e)

        self._tfBuffer = tf2_ros.Buffer()
        self._tfListener = tf2_ros.TransformListener(self._tfBuffer)
        self._tfBr = tf2_ros.TransformBroadcaster()

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
        self._action_server.start()

        rospy.loginfo("Started pose command action server")
        rospy.loginfo("Found all estimation services")

    def preempt_callback(self):
        """
        Preempts the current goal with a new goal
        """
        rospy.loginfo("Current state preempted!")
        self._action_server.set_preempted()

    def goal_callback(self):
        """
        Accepts a goal from a client, and tries to execute the action to its
        completion or until its interuption (pre-empt)
        """
        # accepts a goal from the client
        accepted_goal = self._action_server.accept_new_goal()
        self.use_estimation = accepted_goal.pose.use_estimation
        X_Pregrasp = from_ros_pose(accepted_goal.pose.pose)

        # get current pose
        try:
            tf_X_EE = self._tfBuffer.lookup_transform(
                'panda_link0',
                'panda_EE',
                rospy.Time())
            X_Gcurr = from_ros_transform(tf_X_EE.transform)
            # rospy.loginfo("drake translation: {}".format(
            #     X_Pregrasp.translation()))
            # rospy.loginfo("tf translation: {}".format(
            #     np.array([t.x, t.y, t.z])))
            horizon = np.linalg.norm(
                X_Pregrasp.translation() -
                X_Gcurr.translation()) / self._velocity
            rospy.loginfo("time horizon: {}".format(horizon))
            X_G = {}
            times = {}
            X_G["initial"] = X_Gcurr
            X_G["final"] = X_Pregrasp
            times["initial"] = 0.0
            times["final"] = horizon
            self._plan = MakePegPoseTrajectory(X_G, times)

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return

        self._idx = 0
        if self.use_estimation:
            try:
                # change the stiffness parameters
                self.client.update_configuration(
                    {"translational_stiffness": 720})
                self.client.update_configuration(
                    {"translational_damping": 32})
                self.client.update_configuration(
                    {"rotational_stiffness": 45})
                self.client.update_configuration(
                    {"rotational_damping": 2.1})
                success = self._start_estimation_srv()
            except Exception as e:
                rospy.logwarn("Service call failed: %s", e)
        else:
            # keep the stiffness params
            self.client.update_configuration(
                {"translational_stiffness": 2000})
            self.client.update_configuration(
                {"translational_damping": 89})
            self.client.update_configuration(
                {"rotational_stiffness": 150})
            self.client.update_configuration(
                {"rotational_damping": 7.0})

        self._steps = horizon * self._freq
        dur = rospy.Duration(1.0/self._freq)
        self._timer = rospy.Timer(dur, self.timer_callback)

    def timer_callback(self, event):

        # make sure that the action server is active
        if (not self._action_server.is_active()):
            self._timer.shutdown()
            rospy.logwarn("Action server is not active")
            self._result.reached_goal = False
            self._action_server.set_aborted(self._result)
            return
        if self._plan is None:
            rospy.loginfo("Motion plan has not been set!")
            self._timer.shutdown()
        if self._idx == 0:
            self._plan_start_time = rospy.Time.now().to_sec()
        if self._idx < self._steps:
            self._idx += 1
            time = rospy.Time.now().to_sec() - self._plan_start_time
            X_Gcommand = self._plan.GetPose(time)

            if not self._debug:
                # sends calculated pose to robot
                pose = geom_msg.PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "e_pose"
                pose.pose = to_ros_pose(X_Gcommand)

                self._equilibrium_pose_pub.publish(pose)
            self._feedback.progress = (self._idx * 100)/self._steps
            self._action_server.publish_feedback(self._feedback)
        else:
            self._timer.shutdown()
            rospy.loginfo("Plan succeeded!")
            self._result.reached_goal = True
            if self.use_estimation:
                estimation_stopped = self._stop_estimation_srv()
                estimation_reset = self._reset_estimation_srv()
                self.use_estimation = False
            self._action_server.set_succeeded(self._result)
            return


if __name__ == "__main__":
    rospy.init_node("cmd_pose_action_server", anonymous=True)
    cmd_pose_server = CmdPoseActionServer(rospy.get_name(), False)
    rospy.spin()
