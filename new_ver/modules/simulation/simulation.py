import time
import numpy as np
from math import pi

from modules.model_training.model import Model
from modules.data_collection.data_collection import SimulationDataCollector

from modules.simulation.modules.cam import Cam
from modules.simulation.modules.arm import Arm
from modules.simulation.modules.env import Environment


# only for WSL2, in normal Ubuntu, comment 2 lines below
import matplotlib
matplotlib.use('TkAgg')


count = 0


class Simulation:
    def __init__(self, environment=None, arm=None, cam=None):
        self.environment = environment
        self.arm = arm
        self.cam = cam

    def setup(self):
        self.arm = Arm(base=np.array([3, 0, -0.5]), l1=0.7, l2=0.4)

        self.cam = Cam(self.arm)
        self.cam.set_cam_angle(0, pi/2, pi/2)
        self.environment = Environment(self.cam, self.arm)

    def mode_input(self):
        self.arm.input_pos(unit='rad')
        self.arm.update()
        self.cam.update_arm()

    def mode_random(self):
        self.arm.input_pos(unit='rad')
        self.arm.update()
        self.cam.update_arm()
    def mode_model(self):
        self.arm.input_pos(unit='rad')
        self.arm.update()
        self.cam.update_arm()

    def run(self, visualize=True, mode=None):
        count_loop = 0

        if visualize:
            import matplotlib.pyplot as plt
            plt.ion()
            self.environment.visualize(plt)
            self.cam.visualize(plt)

            # c_arm_plot, = arm.model.ax.plot(
            #     arm.theta[0], arm.theta[1], arm.theta[2], 'bD', markersize=12)
        if not mode:
            while True:
                self.mode_input()

                if visualize:
                    self.environment.update_visualization()
                    self.cam.update_draw()
                    # c_arm_plot.set_data_3d(arm.theta)
                    plt.pause(0.001)
        if mode == 'random':
            while True:
                self.mode_random()

                if visualize:
                    self.environment.update_visualization()
                    self.cam.update_draw()
                    # c_arm_plot.set_data_3d(arm.theta)
                    plt.pause(0.001)

        if mode == 'model':
            while True:
                self.mode_model()

                if visualize:
                    self.environment.update_visualization()
                    self.cam.update_draw()
                    # c_arm_plot.set_data_3d(arm.theta)
                    plt.pause(0.001)

            if self.cam.is_ee_on_boundary():
                self.data_collector.write_data_to_dataframe(self.arm.theta)

            # Add any other conditions or actions you need during simulation

    def handle_danger_situation(self, arm, cam):
        for grid in arm.model.grids:
            nearest = min(grid.point_list, key=lambda point: np.linalg.norm(
                arm.theta - point.pos))
            delta = 0.01
            while cam.is_danger(danger=0.05):
                arm.theta = arm.theta + nearest.e3 * delta
                arm.update()
                cam.update_arm()

    def main(self, visualize=True):
        self.data_collector = SimulationDataCollector(
            cols=['J1_base', 'J2_shoulder', 'J3_elbow'])

        self.setup()

        # model = Model.load("models/model2022-07-28.mdl")
        # arm.add_model(model)

        self.run(visualize)


def main(visualize=True):
    sim = Simulation()
    sim.main(visualize)


# def main():

#     plt.ion()

#     data_collector = SimulationDataCollector(
#         cols=['J1_base', 'J2_shoulder', 'J3_elbow'])

#     arm_fig = plt.figure(figsize=(12, 9))
#     arm = Arm(arm_fig, base=np.array([3, 0, -0.5]), l1=0.7, l2=0.4)

#     cam_fig, cam_ax = plt.subplots()
#     cam = Cam(arm, cam_fig, cam_ax)
#     cam.set_cam_angle(0, pi/2, pi/2)

#     cam.draw_cam()
#     cam.draw_boundary()

#     arm.draw_arm()
#     cam.draw_arm()
#     cam.update_draw()

#     # arm.input_pos_from_csv("thetas27.csv")
#     model = Model.load("models/model2022-07-28.mdl")
#     arm.add_model(model)
#     arm.model.show_model()
#     c_arm_plot, = arm.model.ax.plot(
#         arm.theta[0], arm.theta[1], arm.theta[2], 'bD', markersize=12)

#     count_loop = 0
#     while True:
#         # arm.input_pos(mode='key')
#         # print("Arm's angles {}".format(arm.theta))
#         arm.input_pos(unit='rad')
#         # arm.update_pos_from_csv(count_loop)
#         # arm.random_angles()
#         ########
#         arm.update()
#         #####
#         arm.update_draw()
#         ########
#         cam.update_arm()
#         #####
#         cam.update_draw()
#         # time.sleep(5)

#         c_arm_plot.set_data_3d(arm.theta)

#         count_loop += 1
#         if cam.is_danger():
#             from matplotlib.animation import FuncAnimation
#             ims1 = []
#             ims2 = []
#             ims3 = []
#             for grid in arm.model.grids:
#                 nearest = min(grid.point_list, key=lambda point: np.linalg.norm(
#                     arm.theta-point.pos))
#                 print(nearest)
#                 # push back
#                 delta = 0.01
#                 while cam.is_danger(danger=0.05):
#                     arm.theta = arm.theta+nearest.e3*delta
#                     arm.update()
#                     #####
#                     arm.update_draw()
#                     ########
#                     cam.update_arm()
#                     #####
#                     cam.update_draw()
#                     c_arm_plot.set_data_3d(arm.theta)
#                     # time.sleep(0.1)

#                     ###########
#                     # print(cam.ax.findobj())
#                     # ax = cam.for_ani()
#                     # ims1.append(ax.findobj())

#             # ani1 = FuncAnimation(cam.fig,
#             #                     animate_plot,
#             #                     frames=3,
#             #                     interval=100)
#             # f1 = r"animation.gif"
#             # writergif = animation.PillowWriter(fps=30)
#             # ani1.save(f1, writer=writergif)

#         # if cam.is_ee_on_boundary():
#         #     collect_data.write_data_to_dataframe(arm.theta)
#         #     time.sleep(0.0001)
#         #     count +=1

#         # if count == 2000:
#         #     collect_data.save_dataframe()
#         #     print("collected {} data in loop {}--Done".format(count, count_loop))
#         #     break

#         # plt.show()
