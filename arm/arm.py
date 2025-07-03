import ctypes
import os
import sys
import time
from enum import Enum, IntEnum

# if __name__ == "__main__" or __package__ is None:
#     # 当前文件是直接运行的入口文件 → 使用绝对导入
#     import arm.robotic_arm
#     from arm.robotic_arm import *
# else:
#     # 当前文件是被其他模块导入的 → 使用相对导入
#     from . import robotic_arm
#     from .robotic_arm import *
import arm.robotic_arm
from arm.robotic_arm import *

import json

# copy里面的POSE在我们这里面是Pose

Joints = ctypes.c_float * 6


class my_Arm():
    def __init__(self, wifi=False):
        if not wifi:
            self.arm = Arm(RML63_II, '192.168.1.18')
        else:
            self.arm = Arm(RML63_II, '192.168.33.80')
        # x, y, z = self.arm.Algo_Get_Angle()
        # print("X:" + str(x) + " Y:" + str(y) + " Z:" + str(z))
        # curr_pose = self.arm.Algo_Get_Curr_WorkFrame()
        # print("工作坐标系T：", curr_pose.frame_name.name, curr_pose.pose.position.x, curr_pose.pose.position.y, curr_pose.pose.position.z)
        # print("工作坐标系R：", curr_pose.frame_name.name, curr_pose.pose.euler.rx, curr_pose.pose.euler.ry, curr_pose.pose.euler.rz)

        self.joints = []
        self.pose = Pose()

    def algo_set_joint_min_limit(self, joint_limit = [-178.0, -123.0, -178.0, -178.0, -178.0, -360.0]):
        self.arm.Algo_Set_Joint_Min_Limit(joint_limit = joint_limit)

    def __del__(self):
        self.arm.Arm_Socket_Close()

    def getState(self):
        res = self.arm.Get_Current_Arm_State()
        self.joints = [_ for _ in res[1]]
        joints = [round(self.joints[i], 3) for i in range(6)]
        pose = [round(res[2][i], 5) for i in range(6)]
        # print((joints, pose))
        return joints, pose

    def setJoints(self, joints, v = 30, **kwargs):
        self.arm.Movej_Cmd(joints, v = v, **kwargs)

    def setJoints_CANFD(self, joints, follow):
        # 透传是不过机械臂的算法直接控制关节的运动接口，用于用户自行规划算法，高跟随是机械臂不对路径做优化 下发点位关节就运动
        # 低跟随是机械臂对下发的路径点进行一些插补，保障运行的顺滑
        ret = self.arm.Movej_CANFD(joints, follow)  # 不确定是不是高跟随----低跟随
        return ret

    def forward2Pose(self, joints):  # 不确定 6 joins能否用 能用
        pose = self.arm.Algo_Forward_Kinematics(joints)
        return pose

    def inverseJoints(self, pose, block=1):  # 不确定6joins能否用 能用
        self.getState()
        ret, joints = self.arm.Algo_Inverse_Kinematics(self.joints, pose, block)
        if ret == 0:
            return joints  # [round(joints[i],1) for i in range(6)]
        else:
            return None

    def poseCheck(self, pose):  # 不确定6joins能否用 能用
        if self.inverseJoints(pose, block=1):
            return True
        else:
            return False

    def setPose_P(self, pose, speed=20, curvatureR=0, block=1):  # speed:0~100%
        # use joints space,it can go where it can go. And it need palnner, so it cost a very short time to start move
        ret = self.arm.Movej_P_Cmd(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose(self, pose, speed=10, curvatureR=0, block=1):  # speed:0~100%
        # use Descartes space,it goes along straight line, so sometimes it can't go someplace it can go
        ret = self.arm.Movel_Cmd(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose_CANFD(self, pose, follow):  # speed:0~100%     不确定是否可以6pose
        tag = self.arm.Movep_CANFD(pose, follow)
        return tag

    def setJoint_CANFD(self, joint, follow):
        tag = self.arm.Movej_CANFD(joint, follow, expand=0)
        return tag

    def moveStop(self, block=1):
        ret = self.arm.Move_Stop_Cmd(block)
        return ret

    def movePause(self, block=1):
        ret = self.arm.Move_Pause_Cmd(block)
        return ret

    def moveContinue(self, block=1):
        ret = self.arm.Move_Continue_Cmd(block)
        return ret

    def moveClear(self, block=1):
        self.movePause(block=1)
        ret = self.arm.Clear_Current_Trajectory(self.nsocket, block)
        return ret

    def setGripperPosition(self, position, block=1):
        self.arm.Set_Gripper_Position(position, block)  # position:0-1000



    def gripperRelease(self, speed = 500, block = 1):
        self.arm.Set_Gripper_Release(speed, block)  # release to max-li12t

    def gripperPickOnce(self, force, speed = 500, block = 1):
        self.arm.Set_Gripper_Pick(speed, force, block)  # speed:0-1000, force:50-1000

    def gripperPickKeep(self, force, speed=500, block=1):
        self.arm.Set_Gripper_Pick_On(speed, force, block)

    def getLiftState(self):
        _, res, _, _ = self.arm.Get_Lift_State()
        return res

    def setLiftHeight(self, height, speed=50, block=1):
        self.arm.Set_Lift_Height(height, speed, block)  # height:0-980mm speed:0-100

    def file2traj(self, path):
        # 打开文件
        traj = []
        with open(path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                traj.append(json.loads(line.strip())["point"])
        new_traj = []
        for idx in range(0, len(traj), 300):
            new_point = []
            for sta in traj[idx]:
                new_point.append(sta / 1000)
            new_traj.append(new_point)

        print(new_traj)
        # print(len(new_traj))
        return new_traj

    def MoveCartesianTool_Cmd(self, movelengthx=0, movelengthy=0, movelengthz=0, v=0, block=True, trajectory_connect=0,
                              r=0, m_dev=arm.robotic_arm.RM65):
        joint_currrent = self.getState()[0]
        self.arm.MoveCartesianTool_Cmd(joint_currrent, movelengthx, movelengthy, movelengthz, m_dev, v,
                                       trajectory_connect=trajectory_connect, r=r, block=block)

    def openGripper(self, speed=500, block=True, timeout=5):
        tag = self.arm.Set_Gripper_Release(speed=speed, block=block, timeout=timeout)
        return tag

    def closeGripper(self, mode, position=500, speed=500, force=500, block=True, timeout=5):
        if mode == 'force':
            self.arm.Set_Gripper_Pick(speed, force, block=block, timeout=timeout)
        elif mode == 'position':
            self.arm.Set_Gripper_Position(position, block=block, timeout=timeout)
        else:
            raise Exception("invaild mode")

    def Change_Tool_Frame(self, name, block=True):
        tag = self.arm.Change_Tool_Frame(name, block=block)
        if tag != 0:
            raise Exception('工具坐标系切换失败')
        return tag

    def Get_Tool_Frame(self):
        tag, frame = self.arm.Get_Current_Tool_Frame()
        if tag != 0:
            raise Exception('获取当前坐标系失败')
        return tag, frame


if __name__ == "__main__":
    arm = my_Arm(wifi=False)
