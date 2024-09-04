"""
Common functions for evo_ape and evo_rpe, internal only.
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

import argparse
import logging
import typing
import os
import numpy as np

from evo.core.metrics import PoseRelation, Unit
from evo.core.result import Result
from evo.core.trajectory import PosePath3D, PoseTrajectory3D

logger = logging.getLogger(__name__)

SEP = "-" * 80  # separator line

def load_vio_trajectories(
        args: argparse.Namespace
) -> typing.Tuple[PosePath3D, PosePath3D, PosePath3D, str, str, str]:
    from evo.tools import file_interface

    traj_ref: typing.Union[PosePath3D, PoseTrajectory3D]
    traj_est: typing.Union[PosePath3D, PoseTrajectory3D]

    home_point = file_interface.read_home_point(args.est_dir + "/home_point.txt")

    nav_data = file_interface.read_sf_nav_trajectory_file(args.ref_dir + "/imu.txt", home_point)
    vio_data = file_interface.read_sf_vio_trajectory_file(args.est_dir + "/vio.txt")
    print("nav_data:", nav_data["ts"].shape)
    print("vio_data:", vio_data["ts"].shape)
    T_i_b, T_b_c = file_interface.read_extrinsics(args.ref_dir + "/calib_raw.yaml")
    time_shift = 0.0

    T_b_i = np.linalg.inv(T_i_b)
    aligned_nav_data, aligned_vio_data = file_interface.align_vio(nav_data, vio_data, T_b_i, time_shift)

    traj_ref = PoseTrajectory3D(aligned_nav_data["ned"], np.roll(aligned_nav_data["quat"], 1, axis=1) , aligned_nav_data["ts"])
    traj_est = PoseTrajectory3D(aligned_vio_data["pos"], np.roll(aligned_vio_data["quat"], 1, axis=1) , aligned_vio_data["ts"])
   
    ref_name, est_name = os.path.join(args.ref_dir, "imu.txt"), os.path.join(args.est_dir, "vio.txt")

    print("-------------")
    return traj_ref, traj_est, ref_name, est_name, aligned_nav_data, aligned_vio_data


def load_vland_trajectories(
        args: argparse.Namespace
) -> typing.Tuple[PosePath3D, PosePath3D, PosePath3D, str, str, str]:
    from evo.tools import file_interface

    traj_ref: typing.Union[PosePath3D, PoseTrajectory3D]
    traj_est: typing.Union[PosePath3D, PoseTrajectory3D]

    home_point = file_interface.read_home_point(args.est_dir + "/home_point.txt")

    nav_data = file_interface.read_sf_nav_trajectory_file(args.ref_dir + "/imu.txt", home_point)
    vland_data = file_interface.read_sf_vland_trajectory_file(args.est_dir + "/vland.txt")

    T_i_b, T_b_c = file_interface.read_extrinsics(args.ref_dir + "/calib_raw.yaml")
    time_shift = 0.0

    aligned_nav_data, aligned_vland_data = file_interface.align_vland(nav_data, vland_data, T_b_c, time_shift)

    traj_ref = PoseTrajectory3D(aligned_nav_data["ned"], np.roll(aligned_nav_data["quat"], 1, axis=1) , aligned_nav_data["ts"])
    traj_est = PoseTrajectory3D(aligned_vland_data["pos"], np.roll(aligned_vland_data["quat"], 1, axis=1) , aligned_vland_data["ts"])
   
    ref_name, est_name = os.path.join(args.ref_dir, "imu.txt"), os.path.join(args.est_dir, "vland.txt")

    return traj_ref, traj_est, ref_name, est_name, aligned_nav_data, aligned_vland_data


def load_reloc_trajectories(
        args: argparse.Namespace
) -> typing.Tuple[PosePath3D, PosePath3D, PosePath3D, str, str, str]:
    from evo.tools import file_interface

    traj_ref: typing.Union[PosePath3D, PoseTrajectory3D]
    traj_est: typing.Union[PosePath3D, PoseTrajectory3D]

    home_point = file_interface.read_home_point(args.est_dir + "/home_point.txt")

    nav_data = file_interface.read_sf_nav_trajectory_file(args.ref_dir + "/imu.txt", home_point)
    reloc_data = file_interface.read_sf_reloc_trajectory_file(args.est_dir + "/reloc.txt")

    T_i_b, T_c_b = file_interface.read_extrinsics(args.ref_dir + "/calib_raw.yaml")
    time_shift = 0.0

    aligned_nav_data, aligned_reloc_data = file_interface.align_reloc(nav_data, reloc_data, T_c_b, time_shift)

    traj_ref = PoseTrajectory3D(aligned_nav_data["ned"], np.roll(aligned_nav_data["quat"], 1, axis=1) , aligned_nav_data["ts"])
    traj_est = PoseTrajectory3D(aligned_reloc_data["pos"], np.roll(aligned_reloc_data["quat"], 1, axis=1) , aligned_reloc_data["ts"])
   
    ref_name, est_name = os.path.join(args.ref_dir, "imu.txt"), os.path.join(args.est_dir, "reloc.txt")

    return traj_ref, traj_est, ref_name, est_name, aligned_nav_data, aligned_reloc_data


def load_fusion_trajectories(
        args: argparse.Namespace
) -> typing.Tuple[PosePath3D, PosePath3D, PosePath3D, str, str, str]:
    from evo.tools import file_interface

    traj_ref: typing.Union[PosePath3D, PoseTrajectory3D]
    traj_est: typing.Union[PosePath3D, PoseTrajectory3D]

    home_point = file_interface.read_home_point(args.est_dir + "/home_point.txt")

    nav_data = file_interface.read_sf_nav_trajectory_file(args.ref_dir + "/imu.txt", home_point)
    fusion_data = file_interface.read_sf_fusion_trajectory_file(args.est_dir + "/fusion.txt")

    T_i_b, T_b_c = file_interface.read_extrinsics(args.ref_dir + "/calib_raw.yaml")
    time_shift = 0.0

    aligned_nav_data, aligned_fusion_data = file_interface.align_fusion(nav_data, fusion_data, T_b_c, time_shift)

    traj_ref = PoseTrajectory3D(aligned_nav_data["ned"], np.roll(aligned_nav_data["quat"], 1, axis=1) , aligned_nav_data["ts"])
    traj_est = PoseTrajectory3D(aligned_fusion_data["pos"], np.roll(aligned_fusion_data["quat"], 1, axis=1) , aligned_fusion_data["ts"])
   
    ref_name, est_name = os.path.join(args.ref_dir, "imu.txt"), os.path.join(args.est_dir, "fusion.txt")

    return traj_ref, traj_est, ref_name, est_name, aligned_nav_data, aligned_fusion_data


def load_sfvloc_trajectories(
        args: argparse.Namespace
) -> typing.Tuple[PosePath3D, PosePath3D, PosePath3D, str, str, str]:
    from evo.tools import file_interface

    traj_ref: typing.Union[PosePath3D, PoseTrajectory3D]
    traj_est: typing.Union[PosePath3D, PoseTrajectory3D]

    home_point = file_interface.read_home_point(args.est_dir + "/home_point.txt")

    nav_data = file_interface.read_sf_nav_trajectory_file(args.ref_dir + "/imu.txt", home_point)
    vloc_data = file_interface.read_sf_vloc_trajectory_file(args.est_dir + "/vloc.txt")
    vo_data = file_interface.read_sf_vo_trajectory_file(args.est_dir + "/vo.txt")

    T_i_b, T_b_c = file_interface.read_extrinsics(args.ref_dir + "/calib_raw.yaml")
    time_shift = 0.0

    aligned_nav_data, aligned_vloc_data = file_interface.align_vloc(nav_data, vloc_data, T_i_b, time_shift)

    traj_ref = PoseTrajectory3D(aligned_nav_data["ned"], np.roll(aligned_nav_data["quat"], 1, axis=1) , aligned_nav_data["ts"])
    traj_est = PoseTrajectory3D(aligned_vloc_data["pos"], np.roll(aligned_vloc_data["quat"], 1, axis=1) , aligned_vloc_data["ts"])
    
    ref_name, est_name = os.path.join(args.ref_dir, "imu.txt"), os.path.join(args.est_dir, "vloc.txt")

    return traj_ref, traj_est, ref_name, est_name, aligned_nav_data, aligned_vloc_data, vo_data


def load_trajectories(
    args: argparse.Namespace
) -> typing.Tuple[PosePath3D, PosePath3D, str, str]:
    from evo.tools import file_interface

    traj_ref: typing.Union[PosePath3D, PoseTrajectory3D]
    traj_est: typing.Union[PosePath3D, PoseTrajectory3D]

    if args.subcommand == "tum":
        traj_ref = file_interface.read_tum_trajectory_file(args.ref_file)
        traj_est = file_interface.read_tum_trajectory_file(args.est_file)
        ref_name, est_name = args.ref_file, args.est_file
    elif args.subcommand == "kitti":
        traj_ref = file_interface.read_kitti_poses_file(args.ref_file)
        traj_est = file_interface.read_kitti_poses_file(args.est_file)
        ref_name, est_name = args.ref_file, args.est_file
    elif args.subcommand == "euroc":
        traj_ref = file_interface.read_euroc_csv_trajectory(args.state_gt_csv)
        traj_est = file_interface.read_tum_trajectory_file(args.est_file)
        ref_name, est_name = args.state_gt_csv, args.est_file
    elif args.subcommand == "sfvloc":
        traj_ref = file_interface.read_sf_imu_trajectory_file(args.ref_dir)
        traj_est = file_interface.read_sf_vloc_trajectory_file(args.est_dir)
        ref_name, est_name = os.path.join(args.ref_dir, "imu.txt"), os.path.join(args.est_dir, "vloc.txt")
    elif args.subcommand in ("bag", "bag2"):
        import os
        logger.debug("Opening bag file " + args.bag)
        if not os.path.exists(args.bag):
            raise file_interface.FileInterfaceException(
                "File doesn't exist: {}".format(args.bag))
        if args.subcommand == "bag2":
            from rosbags.rosbag2 import Reader as Rosbag2Reader
            bag = Rosbag2Reader(args.bag)  # type: ignore
        else:
            from rosbags.rosbag1 import Reader as Rosbag1Reader
            bag = Rosbag1Reader(args.bag)  # type: ignore
        try:
            bag.open()
            traj_ref = file_interface.read_bag_trajectory(
                bag, args.ref_topic, cache_tf_tree=True)
            traj_est = file_interface.read_bag_trajectory(
                bag, args.est_topic, cache_tf_tree=True)
            ref_name, est_name = args.ref_topic, args.est_topic
        finally:
            bag.close()
    else:
        raise KeyError("unknown sub-command: {}".format(args.subcommand))

    return traj_ref, traj_est, ref_name, est_name


def get_pose_relation(args: argparse.Namespace) -> PoseRelation:
    if args.pose_relation == "full":
        pose_relation = PoseRelation.full_transformation
    elif args.pose_relation == "rot_part":
        pose_relation = PoseRelation.rotation_part
    elif args.pose_relation == "trans_part":
        pose_relation = PoseRelation.translation_part
    elif args.pose_relation == "angle_deg":
        pose_relation = PoseRelation.rotation_angle_deg
    elif args.pose_relation == "angle_rad":
        pose_relation = PoseRelation.rotation_angle_rad
    elif args.pose_relation == "point_distance":
        pose_relation = PoseRelation.point_distance
    elif args.pose_relation == "point_distance_error_ratio":
        pose_relation = PoseRelation.point_distance_error_ratio
    return pose_relation


def get_delta_unit(args: argparse.Namespace) -> Unit:
    delta_unit = Unit.none
    if args.delta_unit == "f":
        delta_unit = Unit.frames
    elif args.delta_unit == "d":
        delta_unit = Unit.degrees
    elif args.delta_unit == "r":
        delta_unit = Unit.radians
    elif args.delta_unit == "m":
        delta_unit = Unit.meters
    return delta_unit


def plot_reloc_result(args: argparse.Namespace, result: Result, traj_ref: PosePath3D,
                traj_est: PosePath3D,
                nav_data: dict,
                vloc_data: dict,
                traj_ref_full: typing.Optional[PosePath3D] = None) -> None:
    

    from evo.tools import plot
    from evo.tools.settings import SETTINGS

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(SEP)
    logger.debug("Plotting sfvloc results... ")

    if (args.plot_x_dimension == "distances"
            and "distances_from_start" in result.np_arrays):
        x_array = result.np_arrays["distances_from_start"]
        x_label = "$d$ (m)"
    elif (args.plot_x_dimension == "seconds"
          and "seconds_from_start" in result.np_arrays):
        x_array = result.np_arrays["seconds_from_start"]
        x_label = "$t$ (s)"
    elif (args.plot_x_dimension == "original_ts"
          and "original_ts" in result.np_arrays):
        x_array = result.np_arrays["original_ts"]
        x_label = "$t$ (s)"
    else:
        x_array = None
        x_label = "index"


    # # Plot the raw metric values.
    # fig1 = plt.figure(figsize=SETTINGS.plot_figsize)

    # plot.error_array(
    #     fig1.gca(), result.np_arrays["error_array"], x_array=x_array,
    #     statistics={
    #         s: result.stats[s]
    #         for s in SETTINGS.plot_statistics if s not in ("min", "max")
    #     }, name=result.info["label"], title=result.info["title"],
    #     xlabel=x_label)

    plot_collection = plot.PlotCollection(result.info["title"])

    # Pose
    fig_traj_xyz, axarr_traj_xyz = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_rpy, axarr_traj_rpy = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))

    plot_mode = plot.PlotMode[args.plot_mode]
    length_unit = Unit(SETTINGS.plot_trajectory_length_unit)
    ax_traj = plot.prepare_axis(fig_traj_traj, plot_mode,
                                length_unit=length_unit)
    
    plot.traj_xyz(axarr_traj_xyz, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0)
    
    color = next(ax_traj._get_lines.prop_cycler)['color']
    plot.traj_xyz(axarr_traj_xyz, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0)
    
    plot_collection.add_figure("xyz_view", fig_traj_xyz)
    plot_collection.add_figure("rpy_view", fig_traj_rpy)


    ## reloc
    fig_error_xyz, axarr_error_xyz =  plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_xyz, subplots_num=3, ylabels=["x", "y", "z"],
        info_array=result.np_arrays["error_xyz_array"],
        x_array=x_array,
        title="position_error",
        xlabel=x_label)
    
    plot_collection.add_figure("error_xyz", fig_error_xyz)
    



    fig_error_ypr, axarr_error_ypr = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_ypr, subplots_num=3, ylabels=["y", "p", "r"],
        info_array=result.np_arrays["error_ypr_array"],
        x_array=x_array,
        title="rotation_error",
        xlabel=x_label)
    plot_collection.add_figure("error_ypr", fig_error_ypr)


    error_xyz_norm = np.linalg.norm(result.np_arrays["error_xyz_array"], axis=1)
    error_ypr_norm = np.linalg.norm(result.np_arrays["error_ypr_array"], axis=1)
    mean_error_xyz = np.mean(error_xyz_norm)
    max_error_xyz = np.max(error_xyz_norm)
    mean_error_ypr = np.mean(error_ypr_norm)
    max_error_ypr = np.max(error_ypr_norm)
    print("mean_error_xyz: ", mean_error_xyz)
    print("mean_error_ypr: ", mean_error_ypr)
    print("max_error_xyz: ", max_error_xyz)
    print("max_error_ypr: ", max_error_ypr)
    # State
    fig_nav_mode, axarr_nav_mode = plt.subplots(3, sharex="col",
                                            figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_mode, subplots_num=3, ylabels=["navi_mode", "flight_mode", "rtk_yaw"], 
        info_array=np.column_stack((nav_data["navi_mode"], nav_data["flight_mode"], nav_data["rtk_yaw"])), x_array=nav_data["ts"],
        title="nav_state",
        xlabel=x_label)
    plot_collection.add_figure("nav_state", fig_nav_mode)

    fig_nav_vel, axarr_nav_velocity = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_velocity, subplots_num=3, ylabels=["Vx", "Vy", "Vz"], 
        info_array=nav_data["velocity"], x_array=nav_data["ts"],
        title="velocity",
        xlabel=x_label)
    plot_collection.add_figure("nav_velocity", fig_nav_vel)

    fig_nav_reset_count, axarr_nav_reset_count = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_reset_count, subplots_num=3, ylabels=["pos_reset", "alti_reset", "head_reset"],
        info_array=nav_data["reset_count"], x_array=nav_data["ts"],
        title="nav_reset_cnt", 
        xlabel=x_label)
    plot_collection.add_figure("nav_reset_cnt", fig_nav_reset_count)


    # Plot the values color-mapped onto the trajectory.
    if(args.plot_mode == "all"):
        plot_mode = plot.PlotMode("xyz")
        fig_xyz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xz")
        fig_xz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("yz")
        fig_yz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xy")
        fig_xy = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)

        plot_collection.add_figure("fig_xyz", fig_xyz)
        plot_collection.add_figure("fig_xz", fig_xz)
        plot_collection.add_figure("fig_yz", fig_yz)
        plot_collection.add_figure("fig_xy", fig_xy)
    else:
        plot_mode = plot.PlotMode(args.plot_mode)
        fig2 = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_collection.add_figure(args.plot_mode, fig2)




    if args.plot:
        plot_collection.show()
        # state_plot_collection.show()
    if args.save_plot:
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)
    plot_collection.close()


def plot_fusion_result(args: argparse.Namespace, result: Result, traj_ref: PosePath3D,
                traj_est: PosePath3D,
                nav_data: dict,
                vloc_data: dict,
                traj_ref_full: typing.Optional[PosePath3D] = None) -> None:
    

    from evo.tools import plot
    from evo.tools.settings import SETTINGS

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(SEP)
    logger.debug("Plotting sfvloc results... ")

    if (args.plot_x_dimension == "distances"
            and "distances_from_start" in result.np_arrays):
        x_array = result.np_arrays["distances_from_start"]
        x_label = "$d$ (m)"
    elif (args.plot_x_dimension == "seconds"
          and "seconds_from_start" in result.np_arrays):
        x_array = result.np_arrays["seconds_from_start"]
        x_label = "$t$ (s)"
    elif (args.plot_x_dimension == "original_ts"
          and "original_ts" in result.np_arrays):
        x_array = result.np_arrays["original_ts"]
        x_label = "$t$ (s)"
    else:
        x_array = None
        x_label = "index"


    # # Plot the raw metric values.
    # fig1 = plt.figure(figsize=SETTINGS.plot_figsize)

    # plot.error_array(
    #     fig1.gca(), result.np_arrays["error_array"], x_array=x_array,
    #     statistics={
    #         s: result.stats[s]
    #         for s in SETTINGS.plot_statistics if s not in ("min", "max")
    #     }, name=result.info["label"], title=result.info["title"],
    #     xlabel=x_label)

    plot_collection = plot.PlotCollection(result.info["title"])

    # Pose
    fig_traj_xyz, axarr_traj_xyz = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_rpy, axarr_traj_rpy = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))

    plot_mode = plot.PlotMode[args.plot_mode]
    length_unit = Unit(SETTINGS.plot_trajectory_length_unit)
    ax_traj = plot.prepare_axis(fig_traj_traj, plot_mode,
                                length_unit=length_unit)
    
    plot.traj_xyz(axarr_traj_xyz, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0)
    
    color = next(ax_traj._get_lines.prop_cycler)['color']
    plot.traj_xyz(axarr_traj_xyz, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0)
    
    plot_collection.add_figure("xyz_view", fig_traj_xyz)
    plot_collection.add_figure("rpy_view", fig_traj_rpy)


    ## fusion
    fig_error_xyz, axarr_error_xyz =  plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_xyz, subplots_num=3, ylabels=["x", "y", "z"],
        info_array=result.np_arrays["error_xyz_array"],
        x_array=x_array,
        title="position_error",
        xlabel=x_label)
    
    plot_collection.add_figure("error_xyz", fig_error_xyz)


    fig_error_ypr, axarr_error_ypr = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_ypr, subplots_num=3, ylabels=["y", "p", "r"],
        info_array=result.np_arrays["error_ypr_array"],
        x_array=x_array,
        title="rotation_error",
        xlabel=x_label)
    plot_collection.add_figure("error_ypr", fig_error_ypr)


    error_xyz_norm = np.linalg.norm(result.np_arrays["error_xyz_array"], axis=1)
    error_ypr_norm = np.linalg.norm(result.np_arrays["error_ypr_array"], axis=1)
    mean_error_xyz = np.mean(error_xyz_norm)
    max_error_xyz = np.max(error_xyz_norm)
    mean_error_ypr = np.mean(error_ypr_norm)
    max_error_ypr = np.max(error_ypr_norm)
    print("mean_error_xyz: ", mean_error_xyz)
    print("mean_error_ypr: ", mean_error_ypr)
    print("max_error_xyz: ", max_error_xyz)
    print("max_error_ypr: ", max_error_ypr)


    # State
    fig_nav_mode, axarr_nav_mode = plt.subplots(3, sharex="col",
                                            figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_mode, subplots_num=3, ylabels=["navi_mode", "flight_mode", "rtk_yaw"], 
        info_array=np.column_stack((nav_data["navi_mode"], nav_data["flight_mode"], nav_data["rtk_yaw"])), x_array=nav_data["ts"],
        title="nav_state",
        xlabel=x_label)
    plot_collection.add_figure("nav_state", fig_nav_mode)

    fig_nav_vel, axarr_nav_velocity = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_velocity, subplots_num=3, ylabels=["Vx", "Vy", "Vz"], 
        info_array=nav_data["velocity"], x_array=nav_data["ts"],
        title="velocity",
        xlabel=x_label)
    plot_collection.add_figure("nav_velocity", fig_nav_vel)

    fig_nav_reset_count, axarr_nav_reset_count = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_reset_count, subplots_num=3, ylabels=["pos_reset", "alti_reset", "head_reset"],
        info_array=nav_data["reset_count"], x_array=nav_data["ts"],
        title="nav_reset_cnt", 
        xlabel=x_label)
    plot_collection.add_figure("nav_reset_cnt", fig_nav_reset_count)


    # Plot the values color-mapped onto the trajectory.
    if(args.plot_mode == "all"):
        plot_mode = plot.PlotMode("xyz")
        fig_xyz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xz")
        fig_xz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("yz")
        fig_yz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xy")
        fig_xy = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)

        plot_collection.add_figure("fig_xyz", fig_xyz)
        plot_collection.add_figure("fig_xz", fig_xz)
        plot_collection.add_figure("fig_yz", fig_yz)
        plot_collection.add_figure("fig_xy", fig_xy)
    else:
        plot_mode = plot.PlotMode(args.plot_mode)
        fig2 = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_collection.add_figure(args.plot_mode, fig2)




    if args.plot:
        plot_collection.show()
        # state_plot_collection.show()
    if args.save_plot:
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)
    plot_collection.close()




def plot_sfvloc_result(args: argparse.Namespace, result: Result, traj_ref: PosePath3D,
                traj_est: PosePath3D,
                nav_data: dict,
                vloc_data: dict,
                vo_data: dict,
                traj_ref_full: typing.Optional[PosePath3D] = None) -> None:
    

    from evo.tools import plot
    from evo.tools.settings import SETTINGS

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(SEP)
    logger.debug("Plotting sfvloc results... ")

    if (args.plot_x_dimension == "distances"
            and "distances_from_start" in result.np_arrays):
        x_array = result.np_arrays["distances_from_start"]
        x_label = "$d$ (m)"
    elif (args.plot_x_dimension == "seconds"
          and "seconds_from_start" in result.np_arrays):
        x_array = result.np_arrays["seconds_from_start"]
        x_label = "$t$ (s)"
    elif (args.plot_x_dimension == "original_ts"
          and "original_ts" in result.np_arrays):
        x_array = result.np_arrays["original_ts"]
        x_label = "$t$ (s)"
    else:
        x_array = None
        x_label = "index"


    # # Plot the raw metric values.
    # fig1 = plt.figure(figsize=SETTINGS.plot_figsize)

    # plot.error_array(
    #     fig1.gca(), result.np_arrays["error_array"], x_array=x_array,
    #     statistics={
    #         s: result.stats[s]
    #         for s in SETTINGS.plot_statistics if s not in ("min", "max")
    #     }, name=result.info["label"], title=result.info["title"],
    #     xlabel=x_label)

    plot_collection = plot.PlotCollection(result.info["title"])

    # Pose
    fig_traj_xyz, axarr_traj_xyz = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_rpy, axarr_traj_rpy = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))

    plot_mode = plot.PlotMode[args.plot_mode]
    length_unit = Unit(SETTINGS.plot_trajectory_length_unit)
    ax_traj = plot.prepare_axis(fig_traj_traj, plot_mode,
                                length_unit=length_unit)
    
    plot.traj_xyz(axarr_traj_xyz, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0)
    
    color = next(ax_traj._get_lines.prop_cycler)['color']
    plot.traj_xyz(axarr_traj_xyz, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0)
    
    plot_collection.add_figure("xyz_view", fig_traj_xyz)
    plot_collection.add_figure("rpy_view", fig_traj_rpy)


    ## vloc
    fig_error_xyz, axarr_error_xyz =  plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_xyz, subplots_num=3, ylabels=["x", "y", "z"],
        info_array=result.np_arrays["error_xyz_array"],
        x_array=x_array,
        title="position_error",
        xlabel=x_label)
    
    plot_collection.add_figure("error_xyz", fig_error_xyz)


    fig_error_ypr, axarr_error_ypr = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_ypr, subplots_num=3, ylabels=["y", "p", "r"],
        info_array=result.np_arrays["error_ypr_array"],
        x_array=x_array,
        title="rotation_error",
        xlabel=x_label)
    plot_collection.add_figure("error_ypr", fig_error_ypr)


    error_xyz_norm = np.linalg.norm(result.np_arrays["error_xyz_array"], axis=1)
    error_ypr_norm = np.linalg.norm(result.np_arrays["error_ypr_array"], axis=1)
    mean_error_xyz = np.mean(error_xyz_norm)
    max_error_xyz = np.max(error_xyz_norm)
    mean_error_ypr = np.mean(error_ypr_norm)
    max_error_ypr = np.max(error_ypr_norm)
    print("mean_error_xyz: ", mean_error_xyz)
    print("mean_error_ypr: ", mean_error_ypr)
    print("max_error_xyz: ", max_error_xyz)
    print("max_error_ypr: ", max_error_ypr)

    ## vo
    fig_vo_pos, axarr_vo_pos = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_vo_pos, subplots_num=3, ylabels=["x", "y", "z"], 
        info_array=vo_data["pos"], x_array=vo_data["ts"],
        title="vo_pos",
        xlabel=x_label)
    plot_collection.add_figure("vo_xyz", fig_vo_pos)


    fig_vo_euler, axarr_vo_euler = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_vo_euler, subplots_num=3, ylabels=["r", "p", "y"], 
        info_array=vo_data["euler"], x_array=vo_data["ts"],
        title="vo_euler",
        xlabel=x_label)
    plot_collection.add_figure("vo_rpy", fig_vo_euler)


    # State
    fig_vloc_status, axarr_vloc_status = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_vloc_status, subplots_num=3, ylabels=["status", "reset_count", "num_inliers"],
        info_array=np.column_stack((vloc_data["status"], vloc_data["reset_count"], vloc_data["num_inliers"])), x_array=vloc_data["ts"], 
        title="vloc_state",
        xlabel=x_label)
    plot_collection.add_figure("vloc_state", fig_vloc_status)

    fig_vo_cost, axarr_vo_cost = plt.subplots(3, sharex="col",
                                              figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_vo_cost, subplots_num=3, ylabels=["cost_time", "is_kf", "reset_count"],
        info_array=np.column_stack((vo_data["cost_time"], vo_data["is_kf"], vo_data["reset_count"])), x_array=vo_data["ts"],
        title="cost_time", 
        xlabel=x_label)
    plot_collection.add_figure("vo_state", fig_vo_cost)      

    fig_nav_mode, axarr_nav_mode = plt.subplots(3, sharex="col",
                                            figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_mode, subplots_num=3, ylabels=["navi_mode", "flight_mode", "rtk_yaw"], 
        info_array=np.column_stack((nav_data["navi_mode"], nav_data["flight_mode"], nav_data["rtk_yaw"])), x_array=nav_data["ts"],
        title="nav_state",
        xlabel=x_label)
    plot_collection.add_figure("nav_state", fig_nav_mode)

    fig_nav_vel, axarr_nav_velocity = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_velocity, subplots_num=3, ylabels=["Vx", "Vy", "Vz"], 
        info_array=nav_data["velocity"], x_array=nav_data["ts"],
        title="velocity",
        xlabel=x_label)
    plot_collection.add_figure("nav_velocity", fig_nav_vel)

    fig_nav_reset_count, axarr_nav_reset_count = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_reset_count, subplots_num=3, ylabels=["pos_reset", "alti_reset", "head_reset"],
        info_array=nav_data["reset_count"], x_array=nav_data["ts"],
        title="nav_reset_cnt", 
        xlabel=x_label)
    plot_collection.add_figure("nav_reset_cnt", fig_nav_reset_count)

    fig_vloc_height, axarr_vloc_height = plt.subplots(2, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_vloc_height, subplots_num=2, ylabels=["nav_height", "vloc_height"],
        info_array=np.column_stack((nav_data["height"], vloc_data["height"])), x_array=vloc_data["ts"],
        title="height", 
        xlabel=x_label)
    plot_collection.add_figure("height", fig_vloc_height)


    # Plot the values color-mapped onto the trajectory.
    if(args.plot_mode == "all"):
        plot_mode = plot.PlotMode("xyz")
        fig_xyz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xz")
        fig_xz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("yz")
        fig_yz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xy")
        fig_xy = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)

        plot_collection.add_figure("fig_xyz", fig_xyz)
        plot_collection.add_figure("fig_xz", fig_xz)
        plot_collection.add_figure("fig_yz", fig_yz)
        plot_collection.add_figure("fig_xy", fig_xy)
    else:
        plot_mode = plot.PlotMode(args.plot_mode)
        fig2 = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_collection.add_figure(args.plot_mode, fig2)




    if args.plot:
        plot_collection.show()
        # state_plot_collection.show()
    if args.save_plot:
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)
    plot_collection.close()



def plot_vio_result(args: argparse.Namespace, result: Result, traj_ref: PosePath3D,
                traj_est: PosePath3D,
                nav_data: dict,
                vloc_data: dict,
                traj_ref_full: typing.Optional[PosePath3D] = None) -> None:
    

    from evo.tools import plot
    from evo.tools.settings import SETTINGS

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(SEP)
    logger.debug("Plotting sfvloc results... ")

    if (args.plot_x_dimension == "distances"
            and "distances_from_start" in result.np_arrays):
        x_array = result.np_arrays["distances_from_start"]
        x_label = "$d$ (m)"
    elif (args.plot_x_dimension == "seconds"
          and "seconds_from_start" in result.np_arrays):
        x_array = result.np_arrays["seconds_from_start"]
        x_label = "$t$ (s)"
    elif (args.plot_x_dimension == "original_ts"
          and "original_ts" in result.np_arrays):
        x_array = result.np_arrays["original_ts"]
        x_label = "$t$ (s)"
    else:
        x_array = None
        x_label = "index"


    # # Plot the raw metric values.
    # fig1 = plt.figure(figsize=SETTINGS.plot_figsize)

    # plot.error_array(
    #     fig1.gca(), result.np_arrays["error_array"], x_array=x_array,
    #     statistics={
    #         s: result.stats[s]
    #         for s in SETTINGS.plot_statistics if s not in ("min", "max")
    #     }, name=result.info["label"], title=result.info["title"],
    #     xlabel=x_label)

    plot_collection = plot.PlotCollection(result.info["title"])

    # Pose
    fig_traj_xyz, axarr_traj_xyz = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_rpy, axarr_traj_rpy = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))

    plot_mode = plot.PlotMode[args.plot_mode]
    length_unit = Unit(SETTINGS.plot_trajectory_length_unit)
    ax_traj = plot.prepare_axis(fig_traj_traj, plot_mode,
                                length_unit=length_unit)
    
    plot.traj_xyz(axarr_traj_xyz, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0)
    
    color = next(ax_traj._get_lines.prop_cycler)['color']
    plot.traj_xyz(axarr_traj_xyz, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0)
    
    plot_collection.add_figure("xyz_view", fig_traj_xyz)
    plot_collection.add_figure("rpy_view", fig_traj_rpy)


    ## vio
    fig_error_xyz, axarr_error_xyz =  plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_xyz, subplots_num=3, ylabels=["x", "y", "z"],
        info_array=result.np_arrays["error_xyz_array"],
        x_array=x_array,
        title="position_error",
        xlabel=x_label)
    
    plot_collection.add_figure("error_xyz", fig_error_xyz)



    fig_error_ypr, axarr_error_ypr = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    
    plot.sfvloc_state_info(
        axarr=axarr_error_ypr, subplots_num=3, ylabels=["y", "p", "r"],
        info_array=result.np_arrays["error_ypr_array"],
        x_array=x_array,
        title="rotation_error",
        xlabel=x_label)
    plot_collection.add_figure("error_ypr", fig_error_ypr)


    error_xyz_norm = np.linalg.norm(result.np_arrays["error_xyz_array"], axis=1)
    error_ypr_norm = np.linalg.norm(result.np_arrays["error_ypr_array"], axis=1)
    mean_error_xyz = np.mean(error_xyz_norm)
    max_error_xyz = np.max(error_xyz_norm)
    mean_error_ypr = np.mean(error_ypr_norm)
    max_error_ypr = np.max(error_ypr_norm)
    print("mean_error_xyz: ", mean_error_xyz)
    print("mean_error_ypr: ", mean_error_ypr)
    print("max_error_xyz: ", max_error_xyz)
    print("max_error_ypr: ", max_error_ypr)


    # State
    fig_nav_mode, axarr_nav_mode = plt.subplots(3, sharex="col",
                                            figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_mode, subplots_num=3, ylabels=["navi_mode", "flight_mode", "rtk_yaw"], 
        info_array=np.column_stack((nav_data["navi_mode"], nav_data["flight_mode"], nav_data["rtk_yaw"])), x_array=nav_data["ts"],
        title="nav_state",
        xlabel=x_label)
    plot_collection.add_figure("nav_state", fig_nav_mode)

    fig_nav_vel, axarr_nav_velocity = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_velocity, subplots_num=3, ylabels=["Vx", "Vy", "Vz"], 
        info_array=nav_data["velocity"], x_array=nav_data["ts"],
        title="velocity",
        xlabel=x_label)
    plot_collection.add_figure("nav_velocity", fig_nav_vel)

    fig_nav_reset_count, axarr_nav_reset_count = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    plot.sfvloc_state_info(
        axarr=axarr_nav_reset_count, subplots_num=3, ylabels=["pos_reset", "alti_reset", "head_reset"],
        info_array=nav_data["reset_count"], x_array=nav_data["ts"],
        title="nav_reset_cnt", 
        xlabel=x_label)
    plot_collection.add_figure("nav_reset_cnt", fig_nav_reset_count)


    # Plot the values color-mapped onto the trajectory.
    if(args.plot_mode == "all"):
        plot_mode = plot.PlotMode("xyz")
        fig_xyz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xz")
        fig_xz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("yz")
        fig_yz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xy")
        fig_xy = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)

        plot_collection.add_figure("fig_xyz", fig_xyz)
        plot_collection.add_figure("fig_xz", fig_xz)
        plot_collection.add_figure("fig_yz", fig_yz)
        plot_collection.add_figure("fig_xy", fig_xy)
    else:
        plot_mode = plot.PlotMode(args.plot_mode)
        fig2 = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_collection.add_figure(args.plot_mode, fig2)




    if args.plot:
        plot_collection.show()
        # state_plot_collection.show()
    if args.save_plot:
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)
    plot_collection.close()




def plot_sfvo_result(args: argparse.Namespace, result: Result, traj_ref: PosePath3D,
                traj_est: PosePath3D,
                traj_ref_full: typing.Optional[PosePath3D] = None) -> None:
    

    from evo.tools import plot
    from evo.tools.settings import SETTINGS

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(SEP)
    logger.debug("Plotting sfvloc results... ")

    if (args.plot_x_dimension == "distances"
            and "distances_from_start" in result.np_arrays):
        x_array = result.np_arrays["distances_from_start"]
        x_label = "$d$ (m)"
    elif (args.plot_x_dimension == "seconds"
          and "seconds_from_start" in result.np_arrays):
        x_array = result.np_arrays["seconds_from_start"]
        x_label = "$t$ (s)"
    elif (args.plot_x_dimension == "original_ts"
          and "original_ts" in result.np_arrays):
        x_array = result.np_arrays["original_ts"]
        x_label = "$t$ (s)"
    else:
        x_array = None
        x_label = "index"

    # Plot the raw metric values.
    fig1 = plt.figure(figsize=SETTINGS.plot_figsize)

    plot.error_array(
        fig1.gca(), result.np_arrays["error_array"], x_array=x_array,
        statistics={
            s: result.stats[s]
            for s in SETTINGS.plot_statistics if s not in ("min", "max")
        }, name=result.info["label"], title=result.info["title"],
        xlabel=x_label)

    plot_collection = plot.PlotCollection(result.info["title"])
    plot_collection.add_figure("raw", fig1)


    fig_traj_xyz, axarr_traj_xyz = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_rpy, axarr_traj_rpy = plt.subplots(3, sharex="col",
                                        figsize=tuple(SETTINGS.plot_figsize))
    fig_traj_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))

    plot_mode = plot.PlotMode[args.plot_mode]
    length_unit = Unit(SETTINGS.plot_trajectory_length_unit)
    ax_traj = plot.prepare_axis(fig_traj_traj, plot_mode,
                                length_unit=length_unit)
    
    plot.traj_xyz(axarr_traj_xyz, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_ref,
                    style=SETTINGS.plot_reference_linestyle,
                    color=SETTINGS.plot_reference_color,
                    label="ref",
                    alpha=SETTINGS.plot_reference_alpha,
                    start_timestamp=0)
    
    color = next(ax_traj._get_lines.prop_cycler)['color']
    plot.traj_xyz(axarr_traj_xyz, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0, length_unit=length_unit)
    
    plot.traj_rpy(axarr_traj_rpy, traj_est,
                    style=SETTINGS.plot_trajectory_linestyle,
                    color=color,
                    label="est",
                    alpha=SETTINGS.plot_trajectory_alpha,
                    start_timestamp=0)
    
    plot_collection.add_figure("xyz_view", fig_traj_xyz)
    plot_collection.add_figure("rpy_view", fig_traj_rpy)


    # fig_error_xyz, axarr_error_xyz =  plt.subplots(3, sharex="col",
    #                                     figsize=tuple(SETTINGS.plot_figsize))
    
    # plot.sfvloc_state_info(
    #     axarr=axarr_error_xyz, subplots_num=3, ylabels=["x", "y", "z"],
    #     info_array=result.np_arrays["error_xyz_array"],
    #     x_array=x_array,
    #     title="position_error",
    #     xlabel=x_label)
    
    # plot_collection.add_figure("error_xyz", fig_error_xyz)


    # fig_error_rpy, axarr_error_rpy = plt.subplots(3, sharex="col",
    #                                     figsize=tuple(SETTINGS.plot_figsize))
    
    # plot.sfvloc_state_info(
    #     axarr=axarr_error_rpy, subplots_num=3, ylabels=["r", "p", "y"],
    #     info_array=result.np_arrays["error_rpy_array"],
    #     x_array=x_array,
    #     title="rotation_error",
    #     xlabel=x_label)
    # plot_collection.add_figure("error_rpy", fig_error_rpy)


    # Plot the values color-mapped onto the trajectory.
    if(args.plot_mode == "all"):
        plot_mode = plot.PlotMode("xyz")
        fig_xyz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xz")
        fig_xz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("yz")
        fig_yz = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_mode = plot.PlotMode("xy")
        fig_xy = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)

        plot_collection.add_figure("fig_xyz", fig_xyz)
        plot_collection.add_figure("fig_xz", fig_xz)
        plot_collection.add_figure("fig_yz", fig_yz)
        plot_collection.add_figure("fig_xy", fig_xy)
    else:
        plot_mode = plot.PlotMode(args.plot_mode)
        fig2 = plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result)
        plot_collection.add_figure(args.plot_mode, fig2)




    if args.plot:
        plot_collection.show()
        # state_plot_collection.show()
    if args.save_plot:
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)
    plot_collection.close()



def plot_status():
    None


def plot_color_map(args, plot_mode, traj_ref, traj_ref_full, traj_est, result):

    from evo.tools import plot
    from evo.tools.settings import SETTINGS
    import matplotlib.pyplot as plt

    fig2 = plt.figure(figsize=SETTINGS.plot_figsize)

    ax = plot.prepare_axis(
        fig2, plot_mode,
        length_unit=Unit(SETTINGS.plot_trajectory_length_unit))

    plot.traj(ax, plot_mode, traj_ref_full if traj_ref_full else traj_ref,
              style=SETTINGS.plot_reference_linestyle,
              color=SETTINGS.plot_reference_color, label='reference',
              alpha=SETTINGS.plot_reference_alpha,
              plot_start_end_markers=SETTINGS.plot_start_end_markers)
    plot.draw_coordinate_axes(ax, traj_ref, plot_mode,
                              SETTINGS.plot_reference_axis_marker_scale)

    if args.plot_colormap_min is None:
        args.plot_colormap_min = result.stats["min"]
    if args.plot_colormap_max is None:
        args.plot_colormap_max = result.stats["max"]
    if args.plot_colormap_max_percentile is not None:
        args.plot_colormap_max = np.percentile(
            result.np_arrays["error_array"], args.plot_colormap_max_percentile)

    plot.traj_colormap(ax, traj_est, result.np_arrays["error_array"],
                       plot_mode, min_map=args.plot_colormap_min,
                       max_map=args.plot_colormap_max,
                       title=result.info["title"],
                       plot_start_end_markers=SETTINGS.plot_start_end_markers)
    plot.draw_coordinate_axes(ax, traj_est, plot_mode,
                              SETTINGS.plot_axis_marker_scale)
    if args.ros_map_yaml:
        plot.ros_map(ax, args.ros_map_yaml, plot_mode)
    if SETTINGS.plot_pose_correspondences:
        plot.draw_correspondence_edges(
            ax, traj_est, traj_ref, plot_mode,
            style=SETTINGS.plot_pose_correspondences_linestyle,
            color=SETTINGS.plot_reference_color,
            alpha=SETTINGS.plot_reference_alpha)

    fig2.axes.append(ax)
    return fig2



def plot_result(args: argparse.Namespace, result: Result, traj_ref: PosePath3D,
                traj_est: PosePath3D,
                traj_ref_full: typing.Optional[PosePath3D] = None) -> None:
    from evo.tools import plot
    from evo.tools.settings import SETTINGS

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(SEP)
    logger.debug("Plotting results... ")
    plot_mode = plot.PlotMode(args.plot_mode)

    # Plot the raw metric values.
    fig1 = plt.figure(figsize=SETTINGS.plot_figsize)
    if (args.plot_x_dimension == "distances"
            and "distances_from_start" in result.np_arrays):
        x_array = result.np_arrays["distances_from_start"]
        x_label = "$d$ (m)"
    elif (args.plot_x_dimension == "seconds"
          and "seconds_from_start" in result.np_arrays):
        x_array = result.np_arrays["seconds_from_start"]
        x_label = "$t$ (s)"
    elif (args.plot_x_dimension == "original_ts"
          and "original_ts" in result.np_arrays):
        x_array = result.np_arrays["original_ts"]
        x_label = "$t$ (s)"
    else:
        x_array = None
        x_label = "index"

    plot.error_array(
        fig1.gca(), result.np_arrays["error_array"], x_array=x_array,
        statistics={
            s: result.stats[s]
            for s in SETTINGS.plot_statistics if s not in ("min", "max")
        }, name=result.info["label"], title=result.info["title"],
        xlabel=x_label)

    # Plot the values color-mapped onto the trajectory.
    fig2 = plt.figure(figsize=SETTINGS.plot_figsize)
    ax = plot.prepare_axis(
        fig2, plot_mode,
        length_unit=Unit(SETTINGS.plot_trajectory_length_unit))

    plot.traj(ax, plot_mode, traj_ref_full if traj_ref_full else traj_ref,
              style=SETTINGS.plot_reference_linestyle,
              color=SETTINGS.plot_reference_color, label='reference',
              alpha=SETTINGS.plot_reference_alpha,
              plot_start_end_markers=SETTINGS.plot_start_end_markers)
    plot.draw_coordinate_axes(ax, traj_ref, plot_mode,
                              SETTINGS.plot_reference_axis_marker_scale)

    if args.plot_colormap_min is None:
        args.plot_colormap_min = result.stats["min"]
    if args.plot_colormap_max is None:
        args.plot_colormap_max = result.stats["max"]
    if args.plot_colormap_max_percentile is not None:
        args.plot_colormap_max = np.percentile(
            result.np_arrays["error_array"], args.plot_colormap_max_percentile)

    plot.traj_colormap(ax, traj_est, result.np_arrays["error_array"],
                       plot_mode, min_map=args.plot_colormap_min,
                       max_map=args.plot_colormap_max,
                       title=result.info["title"],
                       plot_start_end_markers=SETTINGS.plot_start_end_markers)
    plot.draw_coordinate_axes(ax, traj_est, plot_mode,
                              SETTINGS.plot_axis_marker_scale)
    if args.ros_map_yaml:
        plot.ros_map(ax, args.ros_map_yaml, plot_mode)
    if SETTINGS.plot_pose_correspondences:
        plot.draw_correspondence_edges(
            ax, traj_est, traj_ref, plot_mode,
            style=SETTINGS.plot_pose_correspondences_linestyle,
            color=SETTINGS.plot_reference_color,
            alpha=SETTINGS.plot_reference_alpha)
    fig2.axes.append(ax)

    plot_collection = plot.PlotCollection(result.info["title"])
    plot_collection.add_figure("raw", fig1)
    plot_collection.add_figure("map", fig2)
    if args.plot:
        plot_collection.show()
    if args.save_plot:
        plot_collection.export(args.save_plot,
                               confirm_overwrite=not args.no_warnings)
    if args.serialize_plot:
        logger.debug(SEP)
        plot_collection.serialize(args.serialize_plot,
                                  confirm_overwrite=not args.no_warnings)
    plot_collection.close()
