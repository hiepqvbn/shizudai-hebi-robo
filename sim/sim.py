import time 
import numpy as np
from math import sin, cos, pi, sqrt
import modern_robotics as mr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys  
from pathlib import Path  
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from data_collect.data_collect import DataCollect


EPS = 1*10**(-4)

count=0

# # Rotate transformation matrix
def rotate_transform(thex, they, thez):
    # print("rotate angles: {} {} {}".format(thex, they, thez))
    Rx = np.array([[1,         0,          0, 0],
                   [0, cos(thex), -sin(thex), 0],
                   [0, sin(thex),  cos(thex), 0],
                   [0,         0,          0, 1]])
    Ry = np.array([[ cos(they), 0, sin(they), 0],
                   [         0, 1,         0, 0],
                   [-sin(they), 0, cos(they), 0],
                   [         0, 0,         0, 1]])
    Rz = np.array([[cos(thez), -sin(thez), 0, 0],
                   [sin(thez),  cos(thez), 0, 0],
                   [        0,          0, 1, 0],
                   [        0,          0, 0, 1]])
    
    return Rx@Ry@Rz

# Denavit-Hartenberg Homogeneous transformation matrix
# https://sajidnisar.github.io/posts/python_kinematics_dh
def DH_transform(a, alpha, d, theta):
    return np.array([[cos(theta), -sin(theta)*cos(alpha), sin(alpha)*sin(theta), a*cos(theta)],
                     [sin(theta), cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
                     [0, sin(alpha), cos(alpha), d],
                     [0, 0, 0, 1]])

class Arm(object):
    def __init__(self, base=np.zeros(3, dtype=np.float16), l1=1, l2=1) -> None:
        self._base = base
        self._l1=l1
        self._l2=l2
        self._T_00 = rotate_transform(0,0,0)
        # Init actuator's position
        self._theta = np.zeros(3)
        self._end_effector = np.zeros(3)
        self._elbow = np.zeros(3)
        self.update()

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)

        self.fig.suptitle("3DoF Arm")
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim([-1,self.base[0]+2])
        self.ax.set_ylim([-1,self.base[1]+2])
        self.ax.set_zlim([-1,self.base[2]+2])

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
        self.line1.set_data_3d(linx,liny,linz)
        p1 = self.elbow
        p2 = self.end_effector
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        self.line2.set_data_3d(linx,liny,linz)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # theta = {theta1, theta2, theta3} must be np.array(3)
    def update_DH_matrix(self):
        # Shoulder--link[0]
        self.T_01 = DH_transform(0,pi/2,0,self.theta[0])
        # Elbow--link[1]
        self.T_12 = DH_transform(self.l1,0,0,self.theta[1])
        # EndEffector--link[2]
        self.T_23 = DH_transform(self.l2,0,0,self.theta[2])

        self.T_02 = self.T_01@self.T_12
        self.T_03 = self.T_02@self.T_23

    def update_end_effector(self):
        ee = (self.T_03@np.array([0,0,0,1]))[:3] + self.base
        for i in range(3):
            self._end_effector[i] = ee[i]

    def update_elbow(self):
        ee = (self.T_02@np.array([0,0,0,1]))[:3] + self.base
        for i in range(3):
            self._elbow[i] = ee[i]

    def draw_arm(self):
        self.base_point, = self.ax.plot(self.base[0], self.base[1], self.base[2], 'rs', markersize=7)
        self.elbow_point, = self.ax.plot(self.elbow[0], self.elbow[1], self.elbow[2], 'go', markersize=7)
        self.end_effector_point, = self.ax.plot(self.end_effector[0], self.end_effector[1], self.end_effector[2], 'bx', markersize=7)
        self.line1 = self.match2points(self.base, self.elbow, self.ax, "r-")
        self.line2 = self.match2points(self.end_effector, self.elbow, self.ax, "g-")
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def match2points(self, p1, p2, ax, style):
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        
        line, = ax.plot(linx, liny, linz, style)
        return line

    def input_pos(self, unit='deg'):
        the = input("Input actuators' position(3--degree): ")
        the1, the2, the3 = the.split()
        if unit=='deg':
            self._theta[0] = float(the1)*pi/180
            self._theta[1] = float(the2)*pi/180
            self._theta[2] = float(the3)*pi/180
        elif unit=='rad':
            self._theta[0] = float(the1)
            self._theta[1] = float(the2)
            self._theta[2] = float(the3)

    def random_angles(self):
        ran_angles = np.random.random(3)#*2*pi  ###[0,2pi]
        self._theta[0] = ran_angles[0]*pi           ###[0,pi]
        self._theta[1] = ran_angles[1]*pi/2 + pi/4  ###[pi/4,3pi/4]
        self._theta[2] = ran_angles[2]*pi - pi/2    ###[-pi/2,pi/2]
    
    def input_pos_from_csv(self, csv_file):
        import pandas as pd
        
        self.df = pd.read_csv(csv_file)
        print(self.df.head(3))

    def update_pos_from_csv(self, i):
        self._theta[0] = self.df.iloc[i,1]
        self._theta[1] = self.df.iloc[i,2]
        self._theta[2] = self.df.iloc[i,3]
        print("thete: {}".format(self.theta))
        # time.sleep(0.1)

    def add_model(self, model_name):
        from data_collect.model import Model

        model = Model.load(model_name)
        model.end_effector = self.end_effector

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
    def __init__(self, arm, cam_angle=np.array([0,0,0], dtype=np.float16), w=640, h=480, f=0.5, pos=np.array([0,0,0], dtype=np.float16)) -> None:
        self.w = w*10**(-3)
        self.h = h*10**(-3)
        self.arm = arm

        self._f = f
        self._cam_angle = cam_angle
        self._cam_pos = pos

        self.arm_on_screen = np.zeros((3,2), dtype=np.float16)

        self.set_cam_angle(self.camera_angle[0],self.camera_angle[1],self.camera_angle[2])

        # Transform Arm to screen
        self.update_arm()

        # matplotlib camera screen
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Camera Screen")

        self.ax.set_xlim([-self.w/2, self.w/2])
        self.ax.set_ylim([-self.h/2, self.h/2])

    def update_arm(self):
        # print("rotate matrix {}".format(np.linalg.inv(self.T)))
        base = (np.linalg.inv(self.T)@np.append(self.arm.base,1))[:3]
        elbow = (np.linalg.inv(self.T)@np.append(self.arm.elbow,1))[:3]
        end_effector = (np.linalg.inv(self.T)@np.append(self.arm.end_effector,1))[:3]
        # print("world cooperation --arm base:{}".format(self.arm.base))
        # print("camera cooperation --arm base:{}".format(base))
        self.arm_on_screen[0] = self.screen_xy(base)
        self.arm_on_screen[1] = self.screen_xy(elbow)
        self.arm_on_screen[2] = self.screen_xy(end_effector)
        # print("arm on screen:\n {}".format(self.arm_on_screen))

    def update_draw(self):
        self.base_point.set_data(self.arm_on_screen[0])
        self.elbow_point.set_data(self.arm_on_screen[1])
        self.end_effector_point.set_data(self.arm_on_screen[2])

        p1 = self.arm_on_screen[0]
        p2 = self.arm_on_screen[1]
        linx, liny = [p1[0], p2[0]], [p1[1], p2[1]]
        self.line1.set_data(linx, liny)

        p1 = self.arm_on_screen[1]
        p2 = self.arm_on_screen[2]
        linx, liny = [p1[0], p2[0]], [p1[1], p2[1]]
        self.line2.set_data(linx, liny)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_cam(self):
        self.arm.ax.plot(self.position[0],self.position[1],self.position[2], 'yD', markersize=12)

    def draw_arm(self):
        self.base_point, = self.ax.plot(self.arm_on_screen[0], 'rs')
        self.elbow_point, = self.ax.plot(self.arm_on_screen[1], 'go')
        self.end_effector_point, = self.ax.plot(self.arm_on_screen[2], 'bx')

        p1 = self.arm_on_screen[0]
        p2 = self.arm_on_screen[1]
        linx, liny = [p1[0], p2[0]], [p1[1], p2[1]]
        self.line1, = self.ax.plot(linx, liny, 'r-')

        p1 = self.arm_on_screen[1]
        p2 = self.arm_on_screen[2]
        linx, liny = [p1[0], p2[0]], [p1[1], p2[1]]
        self.line2, = self.ax.plot(linx, liny, 'g-')

    def draw_boundary(self):
        ALPHA = 6
        BETA = 0.8
        x = np.linspace(-self.w/2,self.w/2,10)
        y = ALPHA*x + BETA
        self.ax.plot(x,y, 'y--')

    def screen_xy(self, pos):
        """
        pos is np.array--> shape = (3,) [X, Y, Z]
        :return np.array([x,y])
        """
        x = self.f*pos[0]/pos[2]
        y = self.f*pos[1]/pos[2]

        return np.array([x,y])

    def set_cam_angle(self, thex, they, thez):
        # print("set camera angle ---{}, {}, {}".format(thex, they, thez))
        self.camera_angle[0] = thex
        self.camera_angle[1] = they
        self.camera_angle[2] = thez
        self.update_T()

    def set_cam_pos(self, pos):
        self.position[0] = pos[0]
        self.position[1] = pos[1]
        self.position[2] = pos[2]
        self.update_T()

    def update_T(self):
        # Have to set up np.array dtype = np.float16 because if not sometime it auto convert to np.int32
        self._T = rotate_transform(self.camera_angle[0], self.camera_angle[1], self.camera_angle[2])
        self._T[:,3] = np.append(self.position,1)
        # print("Update homogeneous transformation matrix of camera;\n{}".format(self.T))
        self.update_arm()

    def is_ee_on_boundary(self):
        ALPHA = 6
        BETA = 0.8
        x,y=self.arm_on_screen[2]
        dis=np.abs(ALPHA*x-y+BETA)/sqrt(ALPHA**2+1)
        if dis<EPS and (x<self.w/2 and x>-self.w/2) and (y<self.h/2 and y>-self.h/2):
            print("Arm's EE on the boudary {}".format(self.arm.theta))
            return True
        else:
            return False

    @property
    def position(self):
        return self._cam_pos

    @property
    def f(self):
        return self._f

    @property
    def camera_angle(self):
        return self._cam_angle

    @property
    def T(self):
        """
        Homogeneous transformation matrix of camera
        """
        return self._T



if __name__=="__main__":
    plt.ion()

    collect_data = DataCollect(cols=['J1_base', 'J2_shoulder', 'J3_elbow'],is_sim=True)

    arm = Arm(base=np.array([3,0,-0.5]), l1=0.7, l2=0.4)
    cam = Cam(arm)
    cam.set_cam_angle(0,pi/2,pi/2)
    
    
    cam.draw_cam()
    cam.draw_boundary()

    arm.draw_arm()
    cam.draw_arm()
    cam.update_draw()

    arm.input_pos_from_csv("thetas27.csv")

    count_loop = 0
    while True:
        # arm.input_pos(unit='rad')
        arm.update_pos_from_csv(count_loop)
        # arm.random_angles()
        ########
        arm.update()
        #####
        arm.update_draw()
        ########
        cam.update_arm()
        #####
        cam.update_draw()
        # time.sleep(5)

        count_loop +=1
        # if cam.is_ee_on_boundary():
        #     collect_data.write_data_to_dataframe(arm.theta)
        #     time.sleep(0.0001)
        #     count +=1

        # if count == 2000:
        #     collect_data.save_dataframe()
        #     print("collected {} data in loop {}--Done".format(count, count_loop))
        #     break
        
        # plt.show()