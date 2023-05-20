import matplotlib.pyplot as plt
import time
import numpy as np
from math import pi

# from data_collect.data_collect import DataCollect
from modules.model_training.model import Model

from modules.simulation.modules.cam import Cam
from modules.simulation.modules.arm import Arm


# only for WSL2, in normal Ubuntu, comment 2 lines below
import matplotlib
matplotlib.use('TkAgg')


# import sys
# import os
# # Get the parent directory of the current script (simulation.py)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)

# # Add the parent directory to the Python path
# sys.path.append(parent_dir)

# import sys
# from pathlib import Path
# file = Path(__file__). resolve()
# package_root_directory = file.parents[1]
# sys.path.append(str(package_root_directory))


EPS = 1*10**(-4)

count = 0


def main():

    plt.ion()

    # collect_data = DataCollect(
    #     cols=['J1_base', 'J2_shoulder', 'J3_elbow'], is_sim=True)

    arm_fig = plt.figure(figsize=(12, 9))
    arm = Arm(arm_fig, base=np.array([3, 0, -0.5]), l1=0.7, l2=0.4)

    cam_fig, cam_ax = plt.subplots()
    cam = Cam(arm, cam_fig, cam_ax)
    cam.set_cam_angle(0, pi/2, pi/2)

    cam.draw_cam()
    cam.draw_boundary()

    arm.draw_arm()
    cam.draw_arm()
    cam.update_draw()

    # arm.input_pos_from_csv("thetas27.csv")
    model = Model.load("models/model2022-07-28.mdl")
    arm.add_model(model)
    arm.model.show_model()
    c_arm_plot, = arm.model.ax.plot(
        arm.theta[0], arm.theta[1], arm.theta[2], 'bD', markersize=12)

    count_loop = 0
    while True:
        # arm.input_pos(mode='key')
        # print("Arm's angles {}".format(arm.theta))
        arm.input_pos(unit='rad')
        # arm.update_pos_from_csv(count_loop)
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

        c_arm_plot.set_data_3d(arm.theta)

        count_loop += 1
        if cam.is_danger():
            from matplotlib.animation import FuncAnimation
            ims1 = []
            ims2 = []
            ims3 = []
            for grid in arm.model.grids:
                nearest = min(grid.point_list, key=lambda point: np.linalg.norm(
                    arm.theta-point.pos))
                print(nearest)
                # push back
                delta = 0.01
                while cam.is_danger(danger=0.05):
                    arm.theta = arm.theta+nearest.e3*delta
                    arm.update()
                    #####
                    arm.update_draw()
                    ########
                    cam.update_arm()
                    #####
                    cam.update_draw()
                    c_arm_plot.set_data_3d(arm.theta)
                    # time.sleep(0.1)

                    ###########
                    # print(cam.ax.findobj())
                    # ax = cam.for_ani()
                    # ims1.append(ax.findobj())

            # ani1 = FuncAnimation(cam.fig,
            #                     animate_plot,
            #                     frames=3,
            #                     interval=100)
            # f1 = r"animation.gif"
            # writergif = animation.PillowWriter(fps=30)
            # ani1.save(f1, writer=writergif)

        # if cam.is_ee_on_boundary():
        #     collect_data.write_data_to_dataframe(arm.theta)
        #     time.sleep(0.0001)
        #     count +=1

        # if count == 2000:
        #     collect_data.save_dataframe()
        #     print("collected {} data in loop {}--Done".format(count, count_loop))
        #     break

        # plt.show()


if __name__ == "__main__":
    main()
