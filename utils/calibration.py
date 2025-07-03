import numpy as np
from scipy.spatial.transform import Rotation as R


class Convert():

    def __init__(self, name):
        """
        相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量
        修改变换矩阵的同时需同步修改机械臂工具配置
        """
        # 末端坐标系到相机坐标系的变换矩阵，末端坐标系沿着相机坐标系z轴旋转-90度后与相机坐标系方向一致，因为机械臂示教器设置的末端坐标系方向与相机坐标系不一致，需要额外的旋转
        if name == 'gripper':  # 夹爪
            self.rotation_matrix = np.array([[0, -1, 0], # 沿着z轴旋转-90度
                                             [1, 0, 0],
                                             [0, 0, 1]])
            self.translation_vector = np.array([0.09, -0.032, -0.118])  # [相机纵向， 相机横向（光心到中心），相机深度] 对应相机坐标系的uy ux uz 方向是正下 正右 正前
        elif name == 'xi':
            self.rotation_matrix = np.array([[0, -1, 0],
                                             [1, 0, 0],
                                             [0, 0, 1]])
            self.translation_vector = np.array([0.215, -0.032, -0.163])
        elif name == 'xi_e':
            self.rotation_matrix = np.array([[0, -1, 0],
                                             [1, 0, 0],
                                             [0, 0, 1]])
            self.translation_vector = np.array([0.240, -0.032, -0.240])

        elif name == 'pen_e':
            self.rotation_matrix = np.array([[0, -1, 0],
                                             [1, 0, 0],
                                             [0, 0, 1]])
            self.translation_vector = np.array([0.215, -0.032, -0.220])

        else:
            pass

    ##### 将相机坐标系下的坐标转换到基底坐标系 #####
    def __call__(self, x, y, z, x1, y1, z1, rx, ry, rz):
        """
        我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
        机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）
        """
        # 深度相机识别物体返回的坐标
        obj_camera_coordinates = np.array([x, y, z])
        # 机械臂末端的位姿，单位为弧度
        end_effector_pose = np.array([x1, y1, z1,
                                      rx, ry, rz])
        # 将旋转矩阵和平移向量转换为齐次变换矩阵
        T_end_effector_to_camera = np.eye(4)
        T_end_effector_to_camera[:3, :3] = self.rotation_matrix
        T_end_effector_to_camera[:3, 3] = self.translation_vector
        # 机械臂末端的位姿转换为齐次变换矩阵
        position = end_effector_pose[:3]
        orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
        T_base_to_end_effector = np.eye(4)
        T_base_to_end_effector[:3, :3] = orientation
        T_base_to_end_effector[:3, 3] = position
        # 计算物体相对于机械臂基座的位姿
        obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
        # obj_end_effector_coordinates_homo = np.linalg.inv(T_camera_to_end_effector).dot(obj_camera_coordinates_homo)
        obj_end_effector_coordinates_homo = T_end_effector_to_camera.dot(obj_camera_coordinates_homo)
        # print("物体在末端坐标系下的矩阵：", obj_end_effector_coordinates_homo)
        obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
        obj_base_coordinates = obj_base_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标
        # 计算物体的旋转
        obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(self.rotation_matrix)
        obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)
        # 组合结果
        obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
        obj_base_pose[3:] = rx, ry, rz

        return obj_base_pose

    def get_transfer_matrix(self, cur_pose):
        # 将旋转矩阵和平移向量转换为齐次变换矩阵
        T_end_effector_to_camera = np.eye(4)
        T_end_effector_to_camera[:3, :3] = self.rotation_matrix
        T_end_effector_to_camera[:3, 3] = self.translation_vector
        # 机械臂末端的位姿转换为齐次变换矩阵
        position = cur_pose[:3]
        orientation = R.from_euler('xyz', cur_pose[3:], degrees=False).as_matrix()
        T_base_to_end_effector = np.eye(4)
        T_base_to_end_effector[:3, :3] = orientation
        T_base_to_end_effector[:3, 3] = position

        T_baes_to_camera = T_base_to_end_effector.dot(T_end_effector_to_camera)

        return T_baes_to_camera


if __name__ == "__main__":
    import time
    from arm.arm import my_Arm
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from ultralytics import YOLO
    from camera.camera import Realsense

    from DL_model.yolo.yolo_get_coor import infer_get_center_coor


    cam = Realsense()
    arm = my_Arm(wifi=False)

    convert = Convert('xi')
    arm.Change_Tool_Frame('xi')

    model = YOLO(r'../DL_model/yolo/model_weight/yolo_zhituan.engine')

    while True:

        print('start grab')
        color_image, depth_image, depth_frame = cam.get_frame()

        # 模型推理，获取物体中心坐标
        center_list = infer_get_center_coor(model, color_image, depth_image)
        print("检测到纸团：", center_list)

        if len(center_list) == 0:
            time.sleep(3)

        for i in center_list[:1]:
            # 根据图像坐标系获取相机坐标系坐标
            try:
                ux, uy, uz = cam.get_coordinate_from_pic(i[0], i[1], depth_frame)
            except:
                print("深度获取失败")
                break
            if uz > 1:
                print("距离太远")
                continue

            # 从相机坐标系转换到臂基底坐标系，位姿
            cur_pose = arm.getState()[1]
            pose_in_base = convert(ux, uy, uz, *cur_pose)

            print("初始位姿：", cur_pose)
            print("抓取位姿：", pose_in_base)

            # 夹爪运动到纸团位置
            arm.setPose_P(pose_in_base, speed=10, block=True)

            arm.setPose_P(cur_pose, speed=10, block=True)
