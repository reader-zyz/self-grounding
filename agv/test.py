#!/usr/bin/env python
# coding=utf-8

import socket
import json
import time
import signal
import threading
import sys
import math
import re


def sort_key(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)

class RobotApis(object):
    def __init__(self):
        self.default_addr_ = ("192.168.10.10", 31001)
        self.robot_status_ = None
        self.get_robot_status_thread_ = None
        self.notification_cb_ = None
        self.running_ = True
        signal.signal(signal.SIGINT, self.texit)
        signal.signal(signal.SIGTERM, self.texit)

    def create_socket_client(self, timeout=2):
        c = socket.socket()
        c.settimeout(timeout)
        try:
            c.connect(self.default_addr_)
            return c
        except Exception as e:
            print("[error] failed to connect to %s:%d, %s" % (self.default_addr_[0], self.default_addr_[1], e))
            if e.__str__().find("Connection refused") != -1:
                time.sleep(1)
            return None

    def start_get_robot_status_thread(self):
        if self.get_robot_status_thread_ is None or not self.get_robot_status_thread_.is_alive():
            self.get_robot_status_thread_ = threading.Thread(target=self.get_robot_status_thread)
            self.get_robot_status_thread_.setDaemon(True)
            self.get_robot_status_thread_.start()
        return True 

    def get_robot_status_thread(self):
        print("enter get_robot_status_thread")
        s = None
        rcv_empty_count = 0
        while self.running_:
            if s is None:
                s = self.create_socket_client(2)
                if s is None:
                    time.sleep(1)
                    continue
                else:
                    s.sendall("/api/request_data?topic=robot_status&switch=on&frequency=2".encode())
            # if water restarted, the rcv will be ""
            try:
                rcv = s.recv(10240).decode().strip("\n")
                if rcv == "":
                    rcv_empty_count += 1
                if rcv_empty_count >= 10:
                    rcv_empty_count = 0
                    self.robot_status_ = None
                    s.close()
                    s = None
                    print("water may be restarted")
                    continue

            except Exception as e:
                print("[error] get robot status exception:%s" % e)
                self.robot_status_ = None
                s.close()
                s = None
                continue
            rsps = rcv.split("\n")
            for r in rsps:
                try:
                    j = json.loads(r)
                    if j["type"] == "callback" and j["topic"] == "robot_status":
                        self.robot_status_ = j["results"]
                    elif j["type"] == "notification":
                        log = "[%s] %s" % (j["level"], j["description"])
                        if self.notification_cb_ is not None:
                            self.notification_cb_(log)
                except Exception as e:
                    pass
            time.sleep(0.1)
        print("exit get_robot_status_thread")

    def texit(self, signum, frame):
        print("exit tasks")
        self.running_ = False
        self.cancel_move()
        sys.exit()

    def cancel_move(self):
        c = self.create_socket_client(5)
        if c is None:
            return (False, "failed to connect to server")

        c.sendall("/api/move/cancel".encode())
        try:
            for i in range(2):
                rcv = c.recv(1024).decode().strip("\n")
                rcvs = rcv.split("\n")
                for r in rcvs:
                    j = json.loads(r)
                    if j["type"] != "response":
                        continue
                    if j["command"] != "/api/move/cancel":
                        return False, "script bug"
                    if j["status"] == "OK":
                        return True, ""
                    else:
                        return False, j["error_message"]
        except Exception as e:
            return False, e
        finally:
            c.close()
        return False, "unknown error"

    def move_to_marker(self, marker):
        c = self.create_socket_client(5)
        if c is None:
            return (False, "failed to connect to server")
        c.sendall("/api/move?marker={0}".format(marker).encode())
        try:
            for i in range(2):
                rcv = c.recv(1024).decode().strip("\n")
                rcvs = rcv.split("\n")
                for r in rcvs:
                    j = json.loads(r)
                    if j["type"] != "response":
                        continue
                    if j["command"] != "/api/move":
                        return False, "script bug"
                    if j["status"] == "OK":
                        return True, ""
                    else:
                        return False, j["error_message"]
        except Exception as e:
            return False, e
        finally:
            c.close()
        return False, "unknown error"

    def get_markers(self, floor):
        c = self.create_socket_client(5)
        if c is None:
            return (False, "failed to connect to server")

        c.sendall("/api/markers/query_list?floor={0}".format(floor).encode())
        try:
            rcv = ""
            for i in range(5):
                rcv += c.recv(1024).decode()
                if not "\n" in rcv:
                    continue
                rcvs = rcv.split("\n")
                for r in rcvs:
                    j = json.loads(r)
                    if j["type"] != "response":
                        continue
                    if j["command"] != "/api/markers/query_list":
                        return False, "script bug"
                    if j["status"] == "OK":
                        return True, j['results']
                    else:
                        return False, j["error_message"]
        except Exception as e:
            return False, e
        finally:
            c.close()
        return False, "unknown error"

    def normalize(self, z):
        return math.atan2(math.sin(z),math.cos(z))

    def angle_diff(self, a, b):
        a = self.normalize(a)
        b = self.normalize(b)
        d1 = b-a
        d2 = 2*math.pi - math.fabs(d1)
        if d1 > 0:
          d2 *= -1.0
        if math.fabs(d1) < math.fabs(d2):
          return d1
        else:
          return d2

    def move_accuracy_tes(self, marker, prop):
        self.robot_status_ = None
        ret, msg = self.move_to_marker(marker)
        if not ret:
            print("failed to move to marker: {0}, msg:{1}".format(marker, msg))
            return
        time.sleep(2)
        while self.running_:
            if self.robot_status_ is None:
                # get robot status timeout or software restarted or tcp server restarted
                print("[error] can't catch current robot status!")
                time.sleep(1)
                continue
            if self.robot_status_["move_status"] == "running":
                time.sleep(1)
                continue
            elif self.robot_status_["move_status"] == "canceled":
                print("target: {0} canceled".format(marker))
                return
            else:
                cx = self.robot_status_['current_pose']['x']
                cy = self.robot_status_['current_pose']['y']
                cth = self.robot_status_['current_pose']['theta']
                tx = prop['pose']['position']['x']
                ty = prop['pose']['position']['y']
                def get_yaw(pose):
                    z = pose['pose']['orientation']['z']
                    w = pose['pose']['orientation']['w']
                    return self.normalize(2*math.atan2(z, w))
                tth = get_yaw(prop)
                dtrans = math.hypot(cx-tx, cy-ty)
                dth = self.angle_diff(cth, tth)
                dx = (cx-tx)*math.cos(tth)+(cy-ty)*math.sin(tth)
                dy = math.sqrt(dtrans**2-dx**2)
                if dtrans > 0.05 or math.fabs(dth) > 0.1:
                    print("\033[1;31m marker {0} failed, big error: {1} {2} {3},{4}\033[0m".format(marker, dx, dy, dtrans, dth/math.pi*180))
                else:
                    print("\033[1;32m marker {0} succeed, error: {1} {2} {3},{4}\033[0m".format(marker, dx, dy, dtrans, dth/math.pi*180))
                break


if __name__ == '__main__':
    api = RobotApis()
    api.start_get_robot_status_thread()
    # for i in range(10):
    #     print(f'{i} time')
    #     api.move_to_marker('p2')
    api.move_to_marker('a2')

    # suc, markers = api.get_markers(5)
    # if suc:
    #     for key in sorted(markers.keys(), key=sort_key):
    #         api.move_accuracy_tes(key, markers[key])
    # else:
    #     print("failed to fetch markers: {0}".format(markers))

