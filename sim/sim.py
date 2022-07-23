import numpy as np
from math import sin, cos, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Denavit-Hartenberg Homogeneous transformation matrix
def DH_transform(a, alpha, d, theta):
    return np.array([[cos(theta), -sin(theta), 0, a],
                     [cos(alpha)*sin(theta), cos(alpha)*cos(theta), -sin(alpha), -d*sin(alpha)],
                     [sin(alpha)*sin(theta), -sin(alpha)*cos(theta), cos(alpha), d*cos(alpha)],
                     [0, 0, 0, 1]])

class Arm(object):
    def __init__(self, base=np.zeros(3), l1=1, l2=1) -> None:
        self._base = base
        self._l1=l1
        self._l2=l2
        # Init actuator's position
        self._theta = np.zeros(3)
        self.update()

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)

        self.ax.set_xlabel('J1 Base')
        self.ax.set_ylabel('J2 Shoudler')
        self.ax.set_zlabel('J3 Elbow')

        self.ax.set_xlim([-2,2])
        self.ax.set_ylim([-2,2])
        self.ax.set_zlim([-2,2])

    def update(self):
        self.update_DH_matrix()
        self.update_end_effector()
        self.update_elbow()

    # theta = {theta1, theta2, theta3} must be np.array(3)
    def update_DH_matrix(self):
        # Base--link[0]
        self.T_01 = DH_transform(0,0,0,self.theta[0])
        # Shoulder--link[1]
        self.T_12 = DH_transform(self.l1,pi/2,0,self.theta[1])
        # Elbow--link[2]
        self.T_23 = DH_transform(self.l2,0,0,self.theta[2])

        self.T_02 = self.T_01@self.T_12
        self.T_03 = self.T_01@self.T_12@self.T_23

    def update_end_effector(self):
        self._end_effector = (self.T_03@np.array([0,0,0,1]))[:3]+self.base

    def update_elbow(self):
        self._elbow = (self.T_02@np.array([0,0,0,1]))[:3]+self.base

    def draw_arm(self):
       
        self.line1 = self.match2points(self.base, self.elbow, self.ax, "r-")
        self.line2 = self.match2points(self.end_effector, self.elbow, self.ax, "g-")
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def match2points(self, p1, p2, ax, style):
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        
        line, = ax.plot(linx, liny, linz, style)
        return line

    def input_pos(self):
        the = input("Input actuators' position(3--degree): ")
        the1, the2, the3 = the.split()
        self._theta[0] = float(the1)*pi/180
        self._theta[1] = float(the2)*pi/180
        self._theta[2] = float(the3)*pi/180

    @property
    def base(self):
        return self._base
    
    @property
    def end_effector(self):
        return self._end_effector

    @property
    def elbow(self):
        return self._elbow

    @property
    def l1(self):
        return self._l1

    @property
    def l2(self):
        return self._l2

    @property
    def theta(self):
        return self._theta

class Cam(object):
    def __init__(self) -> None:
        pass

if __name__=="__main__":
    plt.ion()
    arm = Arm()
    arm.draw_arm()
    while True:
        arm.input_pos()
        arm.update()
        arm.fig.canvas.draw()
        arm.fig.canvas.flush_events()
        
    # plt.show()
    # print(DH_transform(1,pi/2,0,-pi))