'''
Author: Chen Shudong chenshudong@realman-robot.com
Date: 2022-10-29 11:50:21
LastEditors: Chen Shudong chenshudong@realman-robot.com
LastEditTime: 2022-10-29 12:47:30
FilePath: /undefined/home/alientek/Desktop/hkx/pythonDemo/python_demo3.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import math
import numpy as np

def getPose_fromT(T):
    px = T[0,3]
    py = T[1,3]
    pz = T[2,3]
    rx = math.atan2(T[2,1], T[2,2])
    ry = math.asin(-T[2,0])
    rz = math.atan2(T[1,0], T[0,0])

    return [px, py, pz, rx, ry, rz]


def getT_fromPose(pose):
    [px, py, pz, rx, ry, rz] = pose
    Rx = np.mat([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.mat([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.mat([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])

    t = np.mat([[px],[py],[pz]])
    R = Rz * Ry * Rx
	
    R_ = np.array(R)
    t_ = np.array(t)
    T_1 = np.append(R_, t_, axis = 1)

    zero = np.mat([0,0,0,1])
    T_2 = np.array(zero)

    T = np.append(T_1, T_2, axis = 0)
    T = np.mat(T)

    return T

# 移动
def endEffectorTranslate(axis, dist, pose):
    [px, py, pz, rx, ry, rz] = pose
    Rx = np.mat([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.mat([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.mat([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz * Ry * Rx

    unitVector = np.mat([[0], [0], [1]])
    T = R*unitVector

    pose_trans = [0.0] * 6
    if axis == "x":
        pose_trans[0]=dist
    if axis == "y":
        pose_trans[1]=dist
    if axis == "z":
        pose_trans[2]=dist
    T_move= getT_fromPose(pose_trans)
   
    T_now = getT_fromPose(pose)
    T_moveTarget = T_now*T_move
    pose = getPose_fromT(T_moveTarget)

    return pose

#旋转
def endEffectorRotate(axis, angle, pose):
    [px, py, pz, rx, ry, rz] = pose
    Rx = np.mat([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.mat([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.mat([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz * Ry * Rx

    unitVector = np.mat([[0], [0], [1]])
    T = R*unitVector

    pose_rotate = [0.0] * 6
    if axis == "x":
        pose_rotate[3]=angle
    if axis == "y":
        pose_rotate[4]=angle
    if axis == "z":
        pose_rotate[5]=angle
        
    T_move= getT_fromPose(pose_rotate)
   
    T_now = getT_fromPose(pose)
    T_moveTarget = T_now*T_move
    pose = getPose_fromT(T_moveTarget)

    return pose

def Rt2T(R,t):
    t = np.array([t]).T
    T = np.concatenate((R,t),axis=1)
    T = np.concatenate((T,np.array([[0,0,0,1]])),axis=0)
    return T
   
def Tinv(T):
    r = T[0:3,0:3]
    t = T[0:3,3]
    rinv = np.linalg.inv(r)
    t = np.matmul(-rinv,t)
    inv = Rt2T(rinv,t)
    return inv

theta = -2/3*np.pi
Rz_theta = np.array([   [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [            0,              0, 1],
                        ])
R_0 = np.array([[ 1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 1],
                ])
T_E2C = Rt2T(Rz_theta,[0.04,0.1,0])
T_C2E = Tinv(T_E2C)

T_E2G = Rt2T(Rz_theta,[0,0,0])
T_G2E = Tinv(T_E2G)

T_G2C = Rt2T(R_0,[0.04,0.1,0])
T_C2G = Tinv(T_G2C)

T_G2F = Rt2T(R_0,[0,0,-0.09])
T_F2G = Tinv(T_G2F)

T_C2F = np.matmul(T_C2G,T_G2F)
origine = np.array([0,0,0,1])

T_F2E =  np.matmul(T_G2E, T_F2G)


print('-----------------------')


if __name__ == "__main__":
    
    # Arm  End  Gripper Camera/Fienger  unit(m)
    pose_EinA = [-0.179, -0.01, 0.609, -0.044, 0.187, 2.703]
    # T_E2A = getT_fromPose(pose_EinA)
    # print(T_E2A)
    # pose_EinA = getPose_fromT(T_E2A)
    # print(pose_EinA)

    # p_E = np.array([0,0,0,1])
    # p_A = np.matmul(T_E2A,p_E)
    # print(p_A)
    
    # theta = 2/3*np.pi
    # Rz_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                   [np.sin(theta),  np.cos(theta), 0],
    #                   [            0,              0, 1],
    #                   ])
    # T_E2G = Rt2T(Rz_theta,[0,0,0])
    # p_E = np.array([0.1,0,0,1])
    # p_G = np.matmul(T_E2G,p_E)
    # print(p_G)
    
    # T_E2C = Rt2T(Rz_theta,[0,-0.1,0])
    # p_E = np.array([10,0,0,1])
    # p_C = np.matmul(T_E2C,p_E)
    # print(p_C)


    # R_0 = np.array([[ 1, 0, 0],
    #                 [ 0, 1, 0],
    #                 [ 0, 0, 1],
    #                 ])
    # T_G2C = Rt2T(R_0,[0,-0.1,0])
    # p_G = np.array([0.1,0,0,1])
    # p_C = np.matmul(T_G2C,p_G)
    # print(p_C)


    # R_0 = np.array([[ 1, 0, 0],
    #                 [ 0, 1, 0],
    #                 [ 0, 0, 1],
    #                 ])
    # T_G2F = Rt2T(R_0,[0,0,-0.125])
    # p_G = np.array([0,0,0,1])
    # p_F = np.matmul(T_G2F,p_G)
    # print(p_F)