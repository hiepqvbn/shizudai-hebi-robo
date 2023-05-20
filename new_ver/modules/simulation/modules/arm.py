import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .utils import rotate_transform, DH_transform
from math import pi


class Arm(object):
    def __init__(self, fig, base=np.zeros(3, dtype=np.float16), l1=1, l2=1) -> None:
        self._base = base
        self._l1 = l1
        self._l2 = l2
        self._T_00 = rotate_transform(0, 0, 0)
        # Init actuator's position
        self._theta = np.zeros(3)
        self._end_effector = np.zeros(3)
        self._elbow = np.zeros(3)
        self.update()

        self.fig = fig
        self.ax = Axes3D(self.fig)

        self.fig.suptitle("3DoF Arm")

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim([-1, self.base[0]+2])
        self.ax.set_ylim([-1, self.base[1]+2])
        self.ax.set_zlim([-1, self.base[2]+2])

    def update(self):
        self.update_DH_matrix()
        self.update_end_effector()
        self.update_elbow()
        # print("theta {}".format(self.theta))
        # print("end effector {}".format(self.end_effector))

    def update_draw(self):
        self.base_point.set_data_3d(self.base)
        self.elbow_point.set_data_3d(self.elbow)
        self.end_effector_point.set_data_3d(self.end_effector)

        p1 = self.base
        p2 = self.elbow
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        self.line1.set_data_3d(linx, liny, linz)
        p1 = self.elbow
        p2 = self.end_effector
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        self.line2.set_data_3d(linx, liny, linz)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # theta = {theta1, theta2, theta3} must be np.array(3)
    def update_DH_matrix(self):
        # Shoulder--link[0]
        self.T_01 = DH_transform(0, pi/2, 0, self.theta[0])
        # Elbow--link[1]
        self.T_12 = DH_transform(self.l1, 0, 0, self.theta[1])
        # EndEffector--link[2]
        self.T_23 = DH_transform(self.l2, 0, 0, self.theta[2])

        self.T_02 = self.T_01@self.T_12
        self.T_03 = self.T_02@self.T_23

    def update_end_effector(self):
        ee = (self.T_03@np.array([0, 0, 0, 1]))[:3] + self.base
        for i in range(3):
            self._end_effector[i] = ee[i]

    def update_elbow(self):
        ee = (self.T_02@np.array([0, 0, 0, 1]))[:3] + self.base
        for i in range(3):
            self._elbow[i] = ee[i]

    def draw_arm(self):
        self.base_point, = self.ax.plot(
            self.base[0], self.base[1], self.base[2], 'rs', markersize=7)
        self.elbow_point, = self.ax.plot(
            self.elbow[0], self.elbow[1], self.elbow[2], 'go', markersize=7)
        self.end_effector_point, = self.ax.plot(
            self.end_effector[0], self.end_effector[1], self.end_effector[2], 'bx', markersize=7)
        self.line1 = self.match2points(self.base, self.elbow, self.ax, "r-")
        self.line2 = self.match2points(
            self.end_effector, self.elbow, self.ax, "g-")
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def match2points(self, p1, p2, ax, style):
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]

        line, = ax.plot(linx, liny, linz, style)
        return line

    def input_pos(self, mode='num', unit='deg'):
        the = input("Input actuators' position(3--degree): ")
        if mode == 'num':
            the1, the2, the3 = the.split()
            if unit == 'deg':
                self._theta[0] = float(the1)*pi/180
                self._theta[1] = float(the2)*pi/180
                self._theta[2] = float(the3)*pi/180
            elif unit == 'rad':
                self._theta[0] = float(the1)
                self._theta[1] = float(the2)
                self._theta[2] = float(the3)
        elif mode == "key":
            delta = 10**(-1)
            delta_theta1 = the.count('q')*delta - the.count('a')*delta
            delta_theta2 = the.count('w')*delta - the.count('s')*delta
            delta_theta3 = the.count('e')*delta - the.count('d')*delta
            self._theta[0] += delta_theta1
            self._theta[1] += delta_theta2
            self._theta[2] += delta_theta3

    def random_angles(self):
        ran_angles = np.random.random(3)  # *2*pi  ###[0,2pi]
        self._theta[0] = ran_angles[0]*pi  # [0,pi]
        self._theta[1] = ran_angles[1]*pi/2 + pi/4  # [pi/4,3pi/4]
        self._theta[2] = ran_angles[2]*pi - pi/2  # [-pi/2,pi/2]

    def input_pos_from_csv(self, csv_file):
        import pandas as pd

        self.df = pd.read_csv(csv_file)
        print(self.df.head(3))

    def update_pos_from_csv(self, i):
        self._theta[0] = self.df.iloc[i, 1]
        self._theta[1] = self.df.iloc[i, 2]
        self._theta[2] = self.df.iloc[i, 3]
        print("thete: {}".format(self.theta))
        # time.sleep(0.1)

    def add_model(self, model):
        

        self.model = model
        # print(model.__dict__)
        self.model.end_effector = self.theta     # Configuration place

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

    @theta.setter
    def theta(self, theta):
        if len(theta) == 3:
            self._theta[0] = theta[0]
            self._theta[1] = theta[1]
            self._theta[2] = theta[2]
        else:
            print("Theta value is not on the right form!!!")
