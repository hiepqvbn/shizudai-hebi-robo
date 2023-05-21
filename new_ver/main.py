from modules.simulation.simulation import main as simulation_main

from modules.data_collection import data_collection
from arm.arm_control import pygame_GUI

import numpy as np
import threading
import sys

argvs = sys.argv


def GUI():
    pyGUI = pygame_GUI.pygameGUI()
    while True:
        pyGUI.step()


def collect_data():
    collector = data_collection.DataCollector()
    if len(argvs) == 3:
        if argvs[2] == 'clear':
            collector.clear_csv()
    if len(argvs) == 2:
        while True:
            try:
                (a, b) = input("Please input 2 value or enter to exit: ").split()
                collector.write_data([int(a), int(b)])
            except:
                break
        collector.save_dataframe()
        print(collector.log_df.head())
        print("Done")


# import pathlib
#     import numpy as np
#     import matplotlib.pyplot as plt

#     class mainGUI(pygame_GUI.pygameGUI):
#         def __init__(self, w=640, h=480) -> None:
#             super().__init__(w, h)

#         # rewrite func for button command
#         def on_button_pressed(self, button):
#             super().on_button_pressed(button)
#             if button.name == 'button_b':
#                 self.controller.set_rumble(0.7, 0.7, 300)
#                 # print(self.robot.group_fbk.position)
#                 collect.write_data_to_csv(self.robot.group_fbk.position)

#     # detect_blue_mark.main()
#     pyGUI = mainGUI()
#     collect = data_collection.DataCollect(cols=pyGUI.robot.names)

#     # camera_thread = threading.Thread(target=detect_blue_mark.main, args=())
#     # camera_thread.start()
#     # rc = realsense_depth.DepthCamera()
#     csvfile = pathlib.Path().absolute()/collect.csv_filename
#     end_effector = np.zeros(3)
#     for i in range(3):
#         end_effector[i] = pyGUI.robot.joint_angles[i]
#     # # # plt.ion()
#     model = Model(end_effector=end_effector)

#     model.train(csv_file=csvfile, iteration=10, show_plot=True)
#     # pyGUI.robot.input_model(model)
#     plt.show()
#     while True:
#         pyGUI.step()

def main():
    # code for running with input from terminal will be written here!!!

    if argvs[1] == 'train':
        from modules.model_training.model import Model
        if len(argvs) >= 3:
            csv_file = argvs[2]
            try:
                iteration = int(argvs[3]) if len(argvs) >= 4 else 10
            except ValueError:
                iteration = 10
            visualize = argvs[3] == 'visualize' or argvs[4] == 'visualize' if len(argvs) >= 4 else False
            model = Model(end_effector=np.zeros(3))
            model.train(csv_file=csv_file, iteration=iteration, show_plot=visualize)
        else:
            print("Insufficient arguments. Please provide the CSV filename.")
    


def run_with_F5_key():
    # write code you want to run when press F5 key in VSCode
    simulation_main()


if __name__ == '__main__':
    if len(argvs) > 1:
        main()
    else:
        run_with_F5_key()
