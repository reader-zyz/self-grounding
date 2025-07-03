import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from typing import Optional, Tuple, Union
import os
from datetime import datetime
from PIL import Image

class Realsense:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        初始化RealSense相机（集成自动曝光/白平衡/内参初始化）
        :param width: 图像宽度
        :param height: 图像高度
        :param fps: 帧率
        """
        # 1. 初始化管道和配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 2. 启动流并获取设备
        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()
        
        # 3. 自动配置传感器参数
        self._configure_sensor_settings(device)
        
        # 4. 初始化内参（丢弃前N帧）
        self.depth_intrinsics = None
        for _ in range(5):  # 丢弃前5帧
            frames = self.pipeline.wait_for_frames()
        self._update_intrinsics()

    def _configure_sensor_settings(self, device: rs.device):
        """配置传感器参数（自动曝光/白平衡）"""
        sensors = device.query_sensors()
        for sensor in sensors:
            if sensor.is_depth_sensor():
                # 深度传感器设置
                sensor.set_option(rs.option.enable_auto_exposure, True)
                # sensor.set_option(rs.option.laser_power, 150)  # 激光功率调节
            elif sensor.is_color_sensor():
                # 彩色传感器设置
                sensor.set_option(rs.option.enable_auto_exposure, True)
                sensor.set_option(rs.option.enable_auto_white_balance, True)
                # sensor.set_option(rs.option.saturation, 50)  # 可选：饱和度调节

    def _update_intrinsics(self):
        """更新相机内参"""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    def get_frames(self, aligned_to_color: bool = True) -> Tuple[np.ndarray, np.ndarray, rs.depth_frame]:
        """
        获取对齐后的彩色图、深度图和深度帧
        :param aligned_to_color: 是否将深度图对齐到彩色图
        :return: (color_image, depth_image, depth_frame)
        """
        frames = self.pipeline.wait_for_frames()
        
        if aligned_to_color:
            align = rs.align(rs.stream.color)
            frames = align.process(frames)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("无法获取帧数据")
        
        # 转换为NumPy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 深度图空洞填充（可选）
        depth_image = self._fill_depth_holes(depth_image)
        
        return color_image, depth_image, depth_frame

    def _fill_depth_holes(self, depth_image: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """使用形态学操作填充深度图空洞"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(depth_image, kernel, iterations=iterations)

    def visualize_stream(self, window_name: str = "RealSense", exit_key: str = 'q'):
        """实时显示RGB-D数据流"""
        try:
            while True:
                color_image, depth_image, _ = self.get_frames()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                combined = np.hstack((color_image, depth_colormap))
                cv2.imshow(window_name, combined)
                
                if cv2.waitKey(1) & 0xFF == ord(exit_key):
                    break
        finally:
            cv2.destroyAllWindows()

    def pixel_to_camera_coordinate(self, x: int, y: int, depth_frame: rs.depth_frame) -> Tuple[float, float, float]:
        """
        将像素坐标转换为相机坐标系下的3D坐标
        :param x: 像素x坐标
        :param y: 像素y坐标
        :param depth_frame: 深度帧对象
        :return: (x, y, z) 相机坐标系下的坐标（单位：米）
        """
        depth = depth_frame.get_distance(x, y)
        if depth <= 0:
            raise ValueError("无效深度值")
        
        return rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth)

    def get_point_cloud(self, depth_image: np.ndarray, mask: Optional[np.ndarray] = None, 
                        visualize: bool = False) -> np.ndarray:
        """
        从深度图生成点云（相机坐标系）
        :param depth_image: 深度图（单位：毫米）
        :param mask: 可选掩码，仅计算掩码区域
        :param visualize: 是否可视化点云
        :return: 点云数组 (H, W, 3)
        """
        depth_meters = depth_image.astype(np.float32) / 1000  # 转换为米
        
        # 生成网格坐标
        rows, cols = depth_meters.shape
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # 转换为相机坐标系
        fx, fy = self.depth_intrinsics.fx, self.depth_intrinsics.fy
        cx, cy = self.depth_intrinsics.ppx, self.depth_intrinsics.ppy
        
        z = depth_meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        point_cloud = np.dstack((x, y, z))  # (H, W, 3)
        
        # 应用掩码
        if mask is not None:
            point_cloud[~mask] = 0
        
        # 可视化
        if visualize:
            self._visualize_point_cloud(point_cloud)
        
        return point_cloud

    def _visualize_point_cloud(self, points: np.ndarray):
        """使用Open3D可视化点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([pcd, coord_frame])

    def __del__(self):
        """析构时自动停止流"""
        self.pipeline.stop()

if __name__ == '__main__':
    cam = Realsense()
    cam.vis_stream()

    img = cam.get_frames()[0]

    clean_directory('vis')
    save_and_pause(img)

