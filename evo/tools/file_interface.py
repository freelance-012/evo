# -*- coding: UTF8 -*-
"""
Low- and high-level read/write functions for different file formats.
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import binascii
import csv
import io
import json
import logging
import os
import typing
import zipfile
import scipy.spatial.transform as transform
import pymap3d

import numpy as np
from rosbags.rosbag1 import (Reader as Rosbag1Reader, Writer as Rosbag1Writer)
from rosbags.rosbag2 import (Reader as Rosbag2Reader, Writer as Rosbag2Writer)
from rosbags.serde import deserialize_cdr, ros1_to_cdr, serialize_cdr
from rosbags.serde.serdes import cdr_to_ros1

from evo import EvoException
import evo.core.lie_algebra as lie
import evo.core.transformations as tr
from evo.core import result
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import user, tf_id

logger = logging.getLogger(__name__)

SUPPORTED_ROS_MSGS = {
    "geometry_msgs/msg/PointStamped", "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/PoseWithCovarianceStamped",
    "geometry_msgs/msg/TransformStamped", "nav_msgs/msg/Odometry"
}


class FileInterfaceException(EvoException):
    pass


def has_utf8_bom(file_path):
    """
    Checks if the given file starts with a UTF8 BOM
    wikipedia.org/wiki/Byte_order_mark
    """
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 3:
        return False
    with open(file_path, 'rb') as f:
        return not int(binascii.hexlify(f.read(3)), 16) ^ 0xEFBBBF


def csv_read_matrix(file_path, delim=',', comment_str="#"):
    """
    directly parse a csv-like file into a matrix
    :param file_path: path of csv file (or file handle)
    :param delim: delimiter character
    :param comment_str: string indicating a comment line to ignore
    :return: 2D list with raw data (string)
    """
    if hasattr(file_path, 'read'):  # if file handle
        generator = (line for line in file_path
                     if not line.startswith(comment_str))
        reader = csv.reader(generator, delimiter=delim)
        mat = [row for row in reader]
    else:
        if not os.path.isfile(file_path):
            raise FileInterfaceException("csv file " + str(file_path) +
                                         " does not exist")
        skip_3_bytes = has_utf8_bom(file_path)
        with open(file_path) as f:
            if skip_3_bytes:
                f.seek(3)
            generator = (line for line in f
                         if not line.startswith(comment_str))
            reader = csv.reader(generator, delimiter=delim)
            mat = [row for row in reader]
    return mat


def read_extrinsics(calib_file_path):
    # TODO

    T_b_i = np.array([
        [1.0, 0.0, 0.0, 0.06],
        [0.0, 1.0, 0.0, -0.22], 
        [0.0, 0.0, 1.0, 0.24], 
        [0.0, 0.0, 0.0, 1.0]
    ])
    T_c_i = np.array([
        [-0.9998775979467093, -0.0006085712922001444, -0.015633897956088386, 0.0749370214709683], 
        [0.0004991215860598248, -0.9999753489295139, 0.00700374263801777, -0.0029486613898941725],
        [-0.015637774840475353, 0.006995082149594015, 0.9998532536446333, 0.0028581994316910853], 
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print(T_b_i)
    T_i_b = np.linalg.inv(T_b_i)
    print(T_i_b)

    T_b_c = np.dot(T_b_i, np.linalg.inv(T_c_i))

    return T_i_b, T_b_c


def transform_to_body(xyz, quat, T_i_b):
    R_i_b = T_i_b[:3, :3]
    P_i_b = T_i_b[:3, 3]

    for i in range(len(quat)):
        R_w_i = transform.Rotation.from_quat(quat[i]).as_matrix()
        P_w_i = xyz[i]
        R_w_b = np.dot(R_w_i, R_i_b)
        P_w_b = np.dot(R_w_i, P_i_b) + P_w_i

        quat[i] = transform.Rotation.from_matrix(R_w_b).as_quat()
        xyz[i] = P_w_b
    


def read_tum_trajectory_file(file_path) -> PoseTrajectory3D:
    """
    parses trajectory file in TUM format (timestamp tx ty tz qx qy qz qw)
    :param file_path: the trajectory file path (or file handle)
    :return: trajectory.PoseTrajectory3D object
    """
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    error_msg = ("TUM trajectory files must have 8 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) != 8):
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)
    stamps = mat[:, 0]  # n x 1
    xyz = mat[:, 1:4]  # n x 3
    quat = mat[:, 4:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    if not hasattr(file_path, 'read'):  # if not file handle
        logger.debug("Loaded {} stamps and poses from: {}".format(
            len(stamps), file_path))
    return PoseTrajectory3D(xyz, quat, stamps)


def write_tum_trajectory_file(file_path, traj: PoseTrajectory3D,
                              confirm_overwrite: bool = False) -> None:
    """
    :param file_path: desired text file for trajectory (string or handle)
    :param traj: trajectory.PoseTrajectory3D
    :param confirm_overwrite: whether to require user interaction
           to overwrite existing files
    """
    if isinstance(file_path, str) and confirm_overwrite:
        if not user.check_and_confirm_overwrite(file_path):
            return
    if not isinstance(traj, PoseTrajectory3D):
        raise FileInterfaceException(
            "trajectory must be a PoseTrajectory3D object")
    stamps = traj.timestamps
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    mat = np.column_stack((stamps, xyz, quat))
    np.savetxt(file_path, mat, delimiter=" ")
    if isinstance(file_path, str):
        logger.info("Trajectory saved to: " + file_path)


def read_kitti_poses_file(file_path) -> PosePath3D:
    """
    parses pose file in KITTI format (first 3 rows of SE(3) matrix per line)
    :param file_path: the trajectory file path (or file handle)
    :return: trajectory.PosePath3D
    """
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    error_msg = ("KITTI pose files must have 12 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) != 12):
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)
    # yapf: disable
    poses = [np.array([[r[0], r[1], r[2], r[3]],
                       [r[4], r[5], r[6], r[7]],
                       [r[8], r[9], r[10], r[11]],
                       [0, 0, 0, 1]]) for r in mat]
    # yapf: enable
    if not hasattr(file_path, 'read'):  # if not file handle
        logger.debug("Loaded {} poses from: {}".format(len(poses), file_path))
    return PosePath3D(poses_se3=poses)


def write_kitti_poses_file(file_path, traj: PosePath3D,
                           confirm_overwrite: bool = False) -> None:
    """
    :param file_path: desired text file for trajectory (string or handle)
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D
    :param confirm_overwrite: whether to require user interaction
           to overwrite existing files
    """
    if isinstance(file_path, str) and confirm_overwrite:
        if not user.check_and_confirm_overwrite(file_path):
            return
    # first 3 rows  of SE(3) matrix flattened
    poses_flat = [p.flatten()[:-4] for p in traj.poses_se3]
    np.savetxt(file_path, poses_flat, delimiter=' ')
    if isinstance(file_path, str):
        logger.info("Poses saved to: " + file_path)

def read_sf_imu_trajectory_file(ref_dir) -> PosePath3D:
    """
    parses trajectory file in sf imu format (ts imu.ts status flight_mode tx ty tz yaw pitch roll vx vy vz reset_count[0] reset_count[1] reset_count[2] latitude longitude altitude altitude_msl height)
    :param ref_dir: the trajectory directory(or file handle)
    :return: trajectory.PoseTrajectory3D object
    """
    imu_file_path = os.path.join(ref_dir, "imu.txt")
    raw_mat = csv_read_matrix(imu_file_path, delim=" ", comment_str="#")
    error_msg = ("sf imu trajectory files must have 21 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) != 21):
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)
    
    home_point_file_path = os.path.join(ref_dir, "home_point.txt")
    home_point_mat = csv_read_matrix(home_point_file_path, delim=' ', comment_str="#")
    error_msg = ("sf home point files must have 3 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    print(home_point_mat)
    if not home_point_mat or (len(home_point_mat) > 0 and len(home_point_mat[0]) != 3):
        raise FileInterfaceException(error_msg)
    try:
        home_point_mat = np.array(home_point_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)

    nav_ts = mat[:, 0]  # n x 1
    nav_status = mat[:, 2].astype(int)
    nav_navigation_mode = np.bitwise_and(nav_status, 15)
    nav_rtk_yaw = np.bitwise_and(nav_status, 22)
    nav_flight_mode = mat[:, 3]
    # nav_pos = mat[:, 4:7]  # n x 3
    nav_euler = mat[:, 7:10]  # n x 3
    nav_velocity = mat[:, 10:13]
    nav_reset_count = mat[:, 13:16]
    nav_geodetic = mat[:, 16:19]
    nav_height = mat[:, 20]
    print(nav_geodetic)
    ned = np.empty((nav_geodetic.shape[0], 3))
    for i in range(len(nav_geodetic)):
        ned[i, :] = np.array(pymap3d.geodetic2ned(nav_geodetic[i, 0], nav_geodetic[i, 1], nav_geodetic[i, 2], home_point_mat[0, 1], home_point_mat[0, 0], home_point_mat[0, 2], ell=pymap3d.Ellipsoid.from_name("wgs84"), deg=True))

    print(ned)

    quat = transform.Rotation.from_euler("ZYX", nav_euler, False).as_quat()
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    print(quat)
    if not hasattr(imu_file_path, 'read'):  # if not file handle
        logger.debug("Loaded {} stamps and poses from: {}".format(
            len(nav_ts), imu_file_path))
    

    return PoseTrajectory3D(ned, quat, nav_ts), nav_ts, nav_navigation_mode, nav_rtk_yaw, nav_flight_mode, nav_velocity, nav_reset_count, nav_height


def read_sf_vloc_trajectory_file(est_dir) -> PosePath3D:
    """
    parses trajectory file in sf vloc format (ts status num_inliers reset_count tx ty tz yaw pitch roll latitude longitude height)
    :param est_dir: the trajectory directory
    :return: trajectory.PoseTrajectory3D object
    """
    vloc_file_path = os.path.join(est_dir, "vloc.txt")
    raw_mat = csv_read_matrix(vloc_file_path, delim=" ", comment_str="#")
    error_msg = ("sf imu trajectory files must have at least 13 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) < 13):
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)
    
    vloc_ts = mat[:, 0]
    vloc_status = mat[:, 1]
    vloc_num_inliers = mat[:, 2]
    vloc_reset_count = mat[:, 3]
    vloc_pos = mat[:, 4:7]  # n x 3
    vloc_euler = mat[:, 7:10]  # n x 3
    vloc_latitude = mat[:, 10]
    vloc_longitude = mat[:, 11]
    vloc_height = mat[:, 12]


    valid_mat = mat[vloc_status > 1]


    vloc_valid_ts = valid_mat[:, 0]  # n x 1
    # vloc_status = valid_mat[:, 1]
    # vloc_num_inliers = valid_mat[:, 2]
    # vloc_reset_count = valid_mat[:, 3]
    vloc_pos = valid_mat[:, 4:7]  # n x 3
    vloc_euler = valid_mat[:, 7:10]  # n x 3
    # vloc_latitude = valid_mat[10]
    # vloc_longitude = valid_mat[11]
    # vloc_height = valid_mat[12]

    quat = transform.Rotation.from_euler("ZYX", vloc_euler, True).as_quat()
    # TODO transform fome imu to body
    calib_file_path = os.path.join(est_dir, "calib_raw.yaml")
    T_i_b, T_b_c = read_extrinsics(calib_file_path)
    transform_to_body(vloc_pos, quat, T_i_b)

    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    if not hasattr(vloc_file_path, 'read'):  # if not file handle
        logger.debug("Loaded {} stamps and poses from: {}".format(
            len(vloc_valid_ts), vloc_file_path))
        
    return PoseTrajectory3D(vloc_pos, quat, vloc_valid_ts), vloc_ts, vloc_status, vloc_num_inliers, vloc_reset_count, vloc_height


def read_sf_vo_trajectory_file(est_dir) -> PosePath3D:
    """
    parses trajectory file in sf vo format (ts num_inliers tx ty tz yaw pitch roll is_keyframe time_cost reset_count)
    :param est_dir: the trajectory directory
    :return: trajectory.PoseTrajectory3D object
    """
    vo_file_path = os.path.join(est_dir, "vo.txt")
    raw_mat = csv_read_matrix(vo_file_path, delim=" ", comment_str="#")
    error_msg = ("sf imu trajectory files must have at least 10 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) < 10):
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)
    stamps = mat[:, 0]  # n x 1
    xyz = mat[:, 2:5]  # n x 3
    ypr = mat[:, 5:8]  # n x 3
    quat = transform.Rotation.from_euler("ZYX", ypr, True).as_quat()
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    print(quat)
    if not hasattr(vo_file_path, 'read'):  # if not file handle
        logger.debug("Loaded {} stamps and poses from: {}".format(
            len(stamps), vo_file_path))
    return PoseTrajectory3D(xyz, quat, stamps)



def read_euroc_csv_trajectory(file_path) -> PoseTrajectory3D:
    """
    parses ground truth trajectory from EuRoC MAV state estimate .csv
    :param file_path: <sequence>/mav0/state_groundtruth_estimate0/data.csv
    :return: trajectory.PoseTrajectory3D object
    """
    raw_mat = csv_read_matrix(file_path, delim=",", comment_str="#")
    error_msg = (
        "EuRoC format ground truth must have at least 8 entries per row "
        "and no trailing delimiter at the end of the rows (comma)")
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) < 8):
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)
    stamps = np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds
    xyz = mat[:, 1:4]  # n x 3
    quat = mat[:, 4:8]  # n x 4
    logger.debug("Loaded {} stamps and poses from: {}".format(
        len(stamps), file_path))
    return PoseTrajectory3D(xyz, quat, stamps)


def _get_xyz_quat_from_transform_stamped(
        msg) -> typing.Tuple[typing.List[float], typing.List[float]]:
    xyz = [
        msg.transform.translation.x, msg.transform.translation.y,
        msg.transform.translation.z
    ]
    quat = [
        msg.transform.rotation.w, msg.transform.rotation.x,
        msg.transform.rotation.y, msg.transform.rotation.z
    ]
    return xyz, quat


def _get_xyz_quat_from_pose_or_odometry_msg(
        msg) -> typing.Tuple[typing.List[float], typing.List[float]]:
    # Make nav_msgs/Odometry behave like geometry_msgs/PoseStamped.
    while not hasattr(msg.pose, 'position') and not hasattr(
            msg.pose, 'orientation'):
        msg = msg.pose
    xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    quat = [
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
        msg.pose.orientation.z
    ]
    return xyz, quat

def _get_xyz_quat_from_point_msg(
        msg) -> typing.Tuple[typing.List[float], typing.List[float]]:
    xyz = [msg.point.x, msg.point.y, msg.point.z]
    # geometry_msgs/PointStamped does not have rotation, add unit quaternion.
    quat = [1., 0., 0., 0.]
    return xyz, quat


def get_supported_topics(
        reader: typing.Union[Rosbag1Reader, Rosbag2Reader]) -> list:
    """
    :param reader: opened bag reader (rosbags.rosbag2 or rosbags.rosbag1)
    :return: list of ROS topics that are supported by this module
    """
    return sorted([
        c.topic for c in reader.connections if c.msgtype in SUPPORTED_ROS_MSGS
    ])


def read_bag_trajectory(reader: typing.Union[Rosbag1Reader,
                                             Rosbag2Reader], topic: str,
                        cache_tf_tree: bool = False) -> PoseTrajectory3D:
    """
    :param reader: opened bag reader (rosbags.rosbag2 or rosbags.rosbag1)
    :param topic: trajectory topic of supported message type,
                  or a TF trajectory ID (e.g.: '/tf:map.base_link' )
    :param cache_tf_tree: cache the tf tree. This speeds up the trajectory
                  reading in case multiple TF trajectories are loaded from
                  the same reader.
    :return: trajectory.PoseTrajectory3D
    """
    if not isinstance(reader, (Rosbag1Reader, Rosbag2Reader)):
        raise FileInterfaceException(
            "reader must be a rosbags.rosbags1.reader.Reader "
            "or rosbags.rosbags2.reader.Reader - "
            "rosbag.Bag() is not supported by evo anymore")

    # TODO: Support TF also with ROS2 bags.
    if tf_id.check_id(topic):
        if isinstance(reader, Rosbag1Reader):
            # Use TfCache instead if it's a TF transform ID.
            from evo.tools import tf_cache
            tf_tree_cache = (tf_cache.instance(reader.__hash__())
                             if cache_tf_tree else tf_cache.TfCache())
            return tf_tree_cache.get_trajectory(reader, identifier=topic)
        else:
            raise FileInterfaceException(
                "TF support for ROS2 bags is not implemented")

    if topic not in reader.topics:
        raise FileInterfaceException("no messages for topic '" + topic +
                                     "' in bag")

    msg_type = reader.topics[topic].msgtype
    if msg_type not in SUPPORTED_ROS_MSGS:
        raise FileInterfaceException(
            "unsupported message type: {}".format(msg_type))

    # Choose appropriate message conversion.
    if msg_type == "geometry_msgs/msg/TransformStamped":
        get_xyz_quat = _get_xyz_quat_from_transform_stamped
    elif msg_type == "geometry_msgs/msg/PointStamped":
        logger.warning(
            "geometry_msgs/PointStamped does not contain rotation, "
            "evo will use unit quaternion. Note that rotation metrics will be "
            "invalid and RPE will only be valid with point_distance metric.")
        get_xyz_quat = _get_xyz_quat_from_point_msg
    else:
        get_xyz_quat = _get_xyz_quat_from_pose_or_odometry_msg

    stamps, xyz, quat = [], [], []

    connections = [c for c in reader.connections if c.topic == topic]
    for connection, _, rawdata in reader.messages(
            connections=connections):  # type: ignore
        if isinstance(reader, Rosbag1Reader):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype),
                                  connection.msgtype)
        else:
            msg = deserialize_cdr(rawdata, connection.msgtype)
        # Use the header timestamps (converted to seconds).
        # Note: msg/stamp is a rosbags type here, not native ROS.
        t = msg.header.stamp
        stamps.append(t.sec + (t.nanosec * 1e-9))
        xyz_t, quat_t = get_xyz_quat(msg)
        xyz.append(xyz_t)
        quat.append(quat_t)

    logger.debug("Loaded {} {} messages of topic: {}".format(
        len(stamps), msg_type, topic))

    # yapf: disable
    (connection, _, rawdata) = list(reader.messages(connections=connections))[0]  # type: ignore
    # yapf: enable
    if isinstance(reader, Rosbag1Reader):
        first_msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype),
                                    connection.msgtype)
    else:
        first_msg = deserialize_cdr(rawdata, connection.msgtype)
    frame_id = first_msg.header.frame_id
    return PoseTrajectory3D(np.array(xyz), np.array(quat), np.array(stamps),
                            meta={"frame_id": frame_id})


def write_bag_trajectory(writer, traj: PoseTrajectory3D, topic_name: str,
                         frame_id: str = "") -> None:
    """
    :param writer: opened bag writer (rosbags.rosbag2 or rosbags.rosbag1)
    :param traj: trajectory.PoseTrajectory3D
    :param topic_name: the desired topic name for the trajectory
    :param frame_id: optional ROS frame_id
    """
    from rosbags.typesys.types import (
        geometry_msgs__msg__PoseStamped as PoseStamped, std_msgs__msg__Header
        as Header, geometry_msgs__msg__Pose as Pose, geometry_msgs__msg__Point
        as Position, geometry_msgs__msg__Quaternion as Quaternion,
        builtin_interfaces__msg__Time as Time)
    if not isinstance(traj, PoseTrajectory3D):
        raise FileInterfaceException(
            "trajectory must be a PoseTrajectory3D object")
    if not isinstance(writer, (Rosbag1Writer, Rosbag2Writer)):
        raise FileInterfaceException(
            "writer must be a rosbags.rosbags1.writer.Writer "
            "or rosbags.rosbags2.writer.Writer - "
            "rosbag.Bag() is not supported by evo anymore")

    msgtype = PoseStamped.__msgtype__
    connection = writer.add_connection(topic_name, msgtype)
    for stamp, xyz, quat in zip(traj.timestamps, traj.positions_xyz,
                                traj.orientations_quat_wxyz):
        sec = int(stamp // 1)
        nanosec = int((stamp - sec) * 1e9)
        time = Time(sec, nanosec)
        header = Header(time, frame_id)
        position = Position(x=xyz[0], y=xyz[1], z=xyz[2])
        quaternion = Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])
        pose = Pose(position, quaternion)
        p = PoseStamped(header, pose)
        serialized_msg = serialize_cdr(p, msgtype)
        if isinstance(writer, Rosbag1Writer):
            serialized_msg = cdr_to_ros1(serialized_msg, msgtype)
        writer.write(connection, int(stamp * 1e9), serialized_msg)
    logger.info("Saved geometry_msgs/PoseStamped topic: " + topic_name)


def save_res_file(zip_path, result_obj: result.Result,
                  confirm_overwrite: bool = False) -> None:
    """
    save results to a zip file that can be deserialized with load_res_file()
    :param zip_path: path to zip file (or file handle)
    :param result_obj: evo.core.result.Result instance
    :param confirm_overwrite: whether to require user interaction
           to overwrite existing files
    """
    if isinstance(zip_path, str):
        logger.debug("Saving results to " + zip_path + "...")
    if confirm_overwrite and not user.check_and_confirm_overwrite(zip_path):
        return
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.writestr("info.json", json.dumps(result_obj.info))
        archive.writestr("stats.json", json.dumps(result_obj.stats))
        for name, array in result_obj.np_arrays.items():
            array_buffer = io.BytesIO()
            np.save(array_buffer, array)
            array_buffer.seek(0)
            archive.writestr("{}.npy".format(name), array_buffer.read())
            array_buffer.close()
        for name, traj in result_obj.trajectories.items():
            traj_buffer = io.StringIO()
            if isinstance(traj, PoseTrajectory3D):
                fmt_suffix = ".tum"
                write_tum_trajectory_file(traj_buffer, traj)
            elif isinstance(traj, PosePath3D):
                fmt_suffix = ".kitti"
                write_kitti_poses_file(traj_buffer, traj)
            else:
                raise FileInterfaceException(
                    "unknown format of trajectory {}".format(name))
            traj_buffer.seek(0)
            archive.writestr("{}{}".format(name, fmt_suffix),
                             traj_buffer.read().encode("utf-8"))
            traj_buffer.close()


def load_res_file(zip_path, load_trajectories: bool = False) -> result.Result:
    """
    load contents of a result .zip file saved with save_res_file(...)
    :param zip_path: path to zip file
    :param load_trajectories: set to True to load also the (backup) trajectories
    :return: evo.core.result.Result instance
    """
    logger.debug("Loading result from {} ...".format(zip_path))
    result_obj = result.Result()
    with zipfile.ZipFile(zip_path, mode='r') as archive:
        file_list = archive.namelist()
        if not {"info.json", "stats.json"} <= set(file_list):
            raise FileInterfaceException(
                "{} is not a valid result file".format(zip_path))
        result_obj.info = json.loads(archive.read("info.json").decode("utf-8"))
        result_obj.stats = json.loads(
            archive.read("stats.json").decode("utf-8"))

        # Compatibility: previous evo versions wrote .npz, although it was .npy
        # In any case, np.load() supports both file formats.
        np_files = [f for f in file_list if f.endswith((".npy", ".npz"))]
        for filename in np_files:
            with io.BytesIO(archive.read(filename)) as array_buffer:
                array = np.load(array_buffer)
                name = os.path.splitext(os.path.basename(filename))[0]
                result_obj.add_np_array(name, array)
        if load_trajectories:
            tum_files = [f for f in file_list if f.endswith(".tum")]
            for filename in tum_files:
                with io.TextIOWrapper(archive.open(filename,
                                                   mode='r')) as traj_buffer:
                    traj = read_tum_trajectory_file(traj_buffer)
                    name = os.path.splitext(os.path.basename(filename))[0]
                    result_obj.add_trajectory(name, traj)
            kitti_files = [f for f in file_list if f.endswith(".kitti")]
            for filename in kitti_files:
                with io.TextIOWrapper(archive.open(filename,
                                                   mode='r')) as path_buffer:
                    path = read_kitti_poses_file(path_buffer)
                    name = os.path.splitext(os.path.basename(filename))[0]
                    result_obj.add_trajectory(name, path)
    return result_obj


def load_transform_json(json_path) -> np.ndarray:
    """
    load a transformation stored in xyz + quaternion format in a .json file,
    optionally with a "scale" field.
    :param json_path: path to the .json file (or file handle)
    :return: t (SE(3) or Sim(3) matrix)
    """
    if hasattr(json_path, "read"):
        data = json.load(json_path)
    else:
        with open(json_path, 'r') as tf_file:
            data = json.load(tf_file)
    keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if not all(key in data for key in keys):
        raise FileInterfaceException(
            "invalid transform file - expected keys " + str(keys))
    xyz = np.array([data["x"], data["y"], data["z"]])
    quat = np.array([data["qw"], data["qx"], data["qy"], data["qz"]])
    scale = 1 if "scale" not in data else data["scale"]
    t = lie.sim3(lie.so3_from_se3(tr.quaternion_matrix(quat)), xyz, scale)
    return t


def load_transform(file_path) -> np.ndarray:
    """
    Load a SE(3) or Sim(3) transformation from either
    - a binary .npy or a text file containing a 4x4 matrix,
      saved with either np.save() or np.savetxt()
    - a JSON file with keys x, y, z, qw, qx, qy, qz (+ scale)
    :return: 4x4 transformation matrix
    """
    if os.path.getsize(file_path) < np.lib.format.MAGIC_LEN:
        raise FileInterfaceException(f"Cannot determine type of {file_path}")

    with open(file_path, "rb") as file_handle:
        header = file_handle.read(np.lib.format.MAGIC_LEN)
        if header.startswith(np.lib.format.MAGIC_PREFIX):
            matrix = np.load(file_path)
        elif header.strip().startswith(b"{"):
            matrix = load_transform_json(file_path)
        else:
            matrix = np.loadtxt(file_path)

    if not matrix.shape == (4, 4) or not lie.is_sim3(matrix):
        raise FileInterfaceException(
            f"{file_path} doesn't contain a valid Sim(3) or SE(3) matrix")
    return matrix
