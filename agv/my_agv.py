#!/usr/bin/env python

import socket
import time
import json
import math
import numpy as np


class Agv():
    def __init__(self):
        ip = "192.168.10.10"
        port = 31001
        sock_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_cli.settimeout(5)  # 底盘5秒连接超时
        serve_addr = (ip, port)

        sock_cli.connect(serve_addr)

        # while True:
        #     try:
        #         sock_cli.connect(serve_addr)
        #         break
        #     except socket.error as e:
        #         print("connect", ip, "failed! Attempt to reconnect...")
        #         time.sleep(2.0)

        print("connect", ip, "succeed!")
        self.sock_cli = sock_cli
        self.serve_addr = serve_addr
        self.markerList = None
        self.mapList = None
        self.currentMap = None
        self.moveTarget = None
        self.moveStatus = None
        self.runnStatus = None
        self.powerPercent = None
        self.currentPose = None

        self.status = None

        command = '/api/software/get_version'
        recvmsg = self.sendMsg(command)
        if recvmsg:
            print('version:', recvmsg['results'])

    def socket_close(self):
        self.sock_cli.close()

    def sendMsg(self, command, msg=''):
        while True:
            try:
                self.sock_cli.sendall((command + msg).encode())
                recvmsg = self.sock_cli.recv(2048)

                recvmsg = recvmsg.decode("utf-8")

                recvmsg = recvmsg.split('\n')[0]  # avoid two line of msg

                try:
                    recvmsg = json.loads(recvmsg)  # avoid invalid input
                except:
                    continue

                if 'command' not in recvmsg or recvmsg[
                    'command'] != command: continue  # # The recemsg may not respond to the input command
                if 'type' not in recvmsg or recvmsg[
                    'type'] != 'response': continue  # # The recemsg may not be the msg respose to the command, see handbook
            except socket.error as e:
                print("send msg error:", str(e))
                self.sock_cli.close()
                self.sock_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock_cli.connect(self.servaddr)
                recvmsg = None
            return recvmsg

    #
    def Goto(self, marker):
        command = '/api/move'
        msg = '?marker=' + marker
        self.sendMsg(command, msg)

    def Goto_Location(self, x, y, theta, block=True):
        command = '/api/move'
        msg = f'?location={x},{y},{theta}'
        self.sendMsg(command, msg)
        if block:
            time.sleep(0.5)
            self.getRobotStatus()
            while not self.moveStatus == 'succeeded':
                time.sleep(0.1)
                self.getRobotStatus()
        return

    def Move(self, linear, angular):
        command = '/api/joy_control'
        msg = '?angular_velocity=' + str(angular) + '&linear_velocity=' + str(linear)
        self.sendMsg(command, msg)

    #
    def moveCancel(self):
        command = '/api/move/cancel'
        self.sendMsg(command)

    def eStop(self, flag):
        command = '/api/estop'
        msg = '?flag' + ('true' if flag else 'false')
        self.sendMsg(command, msg)

    #
    def getMapList(self):
        command = '/api/map/list'
        recvmsg = self.sendMsg(command)
        self.mapList = recvmsg['results']

    def setCurrentMap(self, mapname, floor):
        command = '/api/map/set_current_map'
        msg = 'map_name=' + str(mapname) + '&floor=' + str(floor)
        self.sendMsg(command, msg)

    def getCurrentMap(self):
        command = '/api/map/get_current_map'
        recvmsg = self.sendMsg(command)
        self.currentMap = recvmsg['results']

    def getRobotStatus(self):
        command = '/api/robot_status'
        recvmsg = self.sendMsg(command)
        self.status = recvmsg
        # print(recvmsg)
        moveTarget = recvmsg['results']['move_target']
        moveStatus = recvmsg['results']['move_status']  # idle running succeeded failed canceled
        runnStatus = recvmsg['results']['running_status']
        powerPercent = recvmsg['results']['power_percent']  # 0~100%
        currentPose = recvmsg['results']['current_pose']  # x/m y/m theta/rad
        self.moveTarget = moveTarget
        self.moveStatus = moveStatus
        self.runnStatus = runnStatus
        self.powerPercent = powerPercent
        self.currentPose = currentPose

        return recvmsg

    def arrived(self, marker):
        self.getRobotStatus()
        if self.moveTarget == marker and self.moveStatus == 'succeeded' and self.runnStatus == 'idle':
            return True
        else:
            return False

    def get_target_location(self, x, y, dis, alpha=180):
        def coor_to_angle(x1, y1, x2, y2):  # 将xy坐标的方向转化成阈值为0-2pi的弧度值角度
            delta_y = y2 - y1
            delta_x = x2 - x1
            offset = 0
            # 当分子为0时进行特殊值讨论
            if delta_x == 0:
                if delta_y == 0:
                    return 0
                if delta_y > 0:
                    return 90 / 180 * math.pi
                if delta_y < 0:
                    return 270 / 180 * math.pi
            # 根据符号对角度进行修正
            if delta_x < 0:
                offset = 180
            elif delta_y < 0:
                offset = 360
            return math.atan(delta_y / delta_x) + offset / 180 * math.pi

        # alpha为机械臂base坐标到底盘自身坐标系的旋转角度
        alpha = alpha / 180 * math.pi
        # 底盘base坐标系到底盘自身坐标系的变换矩阵
        cur_loc = self.getRobotStatus()['results']['current_pose']
        print("cur_loc:", cur_loc)
        cur_loc['theta'] += alpha
        T = np.array([[math.cos(cur_loc['theta']), - math.sin(cur_loc['theta']), cur_loc['x']],
                      [math.sin(cur_loc['theta']), math.cos(cur_loc['theta']), cur_loc['y']],
                      [0, 0, 1]])
        # 将目标点转化为齐次坐标，该坐标为目标点在底盘自身坐标系的坐标
        delta_coor = np.array([x, y, 1])
        # 将目标点转化为底盘base坐标系下的坐标
        target_coor = T.dot(delta_coor)
        # 计算底盘停止时的指向角度，该角度方向为起始点到目标点的方向
        target_theta = coor_to_angle(cur_loc['x'], cur_loc['y'], target_coor[0], target_coor[1])

        # 使用move接口时，角度3-6度误差，距离8-20厘米误差
        # 使用速度接口时，至少10毫秒发送一次

        # print("物体与基座距离:", math.sqrt(target_coor[0] ** 2 + target_coor[1] ** 2))

        dis_x = dis * math.cos(target_theta)
        dis_y = dis * math.sin(target_theta)

        target_coor[0] -= dis_x
        target_coor[1] -= dis_y

        # print("物体与基座缩短后的距离:", math.sqrt(target_coor[0] ** 2 + target_coor[1] ** 2))

        return target_coor[0], target_coor[1], target_theta


if __name__ == "__main__":

    agv = Agv()
    agv.Goto('A0')
    if agv.arrived():
        pass

    # while True:
    #     cv2.imshow('tt',cv2.imread('/home/realman/AAProject/11.jpg'))

    #     key = cv2.waitKey(1)
    #     if int(key) == ord('q'):
    #         break
    #     elif int(key) == ord('a'):
    #         print('Move')
    #         agv.Move(0,0.1)# angle & linear

    #     elif int(key) == ord('d'):
    #         print('Move')
    #         agv.Move(0,-0.1)

    #     elif int(key) == ord('s'):
    #         print('Move')
    #         agv.Move(-0.1,0)

    #     elif int(key) == ord('w'):
    #         print('Move')
    #         agv.Move(0.1,0)

    #     elif int(key) == ord('e'):
    #         print('getMapList')
    #         re = agv.getMapList()
    #         print(re)
    #     elif int(key) == ord('r'):
    #         print('getCurrentMap')
    #         re = agv.getCurrentMap()
    #         print(re)
    #     elif int(key) == ord('t'):
    #         print('getCurrentMap')
    #         agv.getRobotStatus()
    #         print(agv.currentPose)
    #         print(agv.moveStatus)
    #         print(agv.powerPercent)
    #     elif int(key) == ord('1'):
    #         print('getCurrentMap')
    #         re = agv.Goto('1')
    #     elif int(key) == ord('2'):
    #         print('getCurrentMap')
    #         agv.Goto('2')

    #     else:
    #         agv.Move(0,0)

    #     time.sleep(0.2)

    agv.socket_close()
