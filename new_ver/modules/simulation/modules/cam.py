import numpy as np
from .utils import rotate_transform
from math import sqrt

EPS = 1*10**(-4)


class Cam(object):
    def __init__(self, arm, cam_angle=np.array([0, 0, 0], dtype=np.float16), w=640, h=480, f=0.5, pos=np.array([0, 0, 0], dtype=np.float16)) -> None:
        self.w = w*10**(-3)
        self.h = h*10**(-3)
        self.arm = arm

        self._f = f
        self._cam_angle = cam_angle
        self._cam_pos = pos

        self.arm_on_screen = np.zeros((3, 2), dtype=np.float16)

        self.set_cam_angle(
            self.camera_angle[0], self.camera_angle[1], self.camera_angle[2])

        # Transform Arm to screen
        self.update_arm()

    def visualize_in_env(self, env_ax):
        self.env_ax = env_ax
        self.draw_cam_in_env()

    def visualize(self, plt):
        self.plt = plt
        # matplotlib camera screen
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_title("Camera Screen")

        self.ax.set_xlim([-self.w/2, self.w/2])
        self.ax.set_ylim([-self.h/2, self.h/2])

        self.draw_arm()
        self.draw_boundary()

    def update_arm(self):
        # print("rotate matrix {}".format(np.linalg.inv(self.T)))
        base = (np.linalg.inv(self.T)@np.append(self.arm.base, 1))[:3]
        elbow = (np.linalg.inv(self.T)@np.append(self.arm.elbow, 1))[:3]
        end_effector = (np.linalg.inv(self.T) @
                        np.append(self.arm.end_effector, 1))[:3]
        # print("world cooperation --arm base:{}".format(self.arm.base))
        # print("camera cooperation --arm base:{}".format(base))
        self.arm_on_screen[0] = self.screen_xy(base)
        self.arm_on_screen[1] = self.screen_xy(elbow)
        self.arm_on_screen[2] = self.screen_xy(end_effector)
        # print(f'{self.arm_on_screen=}')

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

    def draw_cam_in_env(self):
        self.env_ax.plot(
            self.position[0], self.position[1], self.position[2], 'yD', markersize=12)

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
        x = np.linspace(-self.w/2, self.w/2, 10)
        y = ALPHA*x + BETA
        self.ax.plot(x, y, 'r--')
        # Danger line
        BETA1 = -sqrt(ALPHA**2+1)*0.02 + BETA
        x = np.linspace(-self.w/2, self.w/2, 10)
        y = ALPHA*x + BETA1
        self.ax.plot(x, y, 'y--')
        # Safe line
        BETA2 = -sqrt(ALPHA**2+1)*0.05 + BETA
        x = np.linspace(-self.w/2, self.w/2, 10)
        y = ALPHA*x + BETA2
        self.ax.plot(x, y, 'g--')

    def screen_xy(self, pos):
        """
        pos is np.array--> shape = (3,) [X, Y, Z]
        :return np.array([x,y])
        """
        x = self.f*pos[0]/pos[2]
        y = self.f*pos[1]/pos[2]

        return np.array([x, y])

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
        self._T = rotate_transform(
            self.camera_angle[0], self.camera_angle[1], self.camera_angle[2])
        self._T[:, 3] = np.append(self.position, 1)
        # print("Update homogeneous transformation matrix of camera;\n{}".format(self.T))
        self.update_arm()

    def ee_to_boundary(self):
        ALPHA = 6
        BETA = 0.8
        x, y = self.arm_on_screen[2]
        dis = np.abs(ALPHA*x-y+BETA)/sqrt(ALPHA**2+1)
        return dis

    def is_danger(self, danger=0.02):
        x, y = self.arm_on_screen[2]
        dis = self.ee_to_boundary()
        if dis < danger and (x < self.w/2 and x > -self.w/2) and (y < self.h/2 and y > -self.h/2):
            print("Dangerous zone !!!")
            return True
        else:
            return False

    def is_ee_on_boundary(self):
        x, y = self.arm_on_screen[2]
        dis = self.ee_to_boundary()
        if dis < EPS and (x < self.w/2 and x > -self.w/2) and (y < self.h/2 and y > -self.h/2):
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
