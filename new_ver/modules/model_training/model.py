from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import pickle

from modules.visualization.data_visualization import *


class Model():
    SPEED = 0.01

    def __init__(self, C_space_arm_pos) -> None:
        self._C_space_arm_pos = C_space_arm_pos

    def show_model(self, show_gridpoint_comps=True):
        from mpl_toolkits.mplot3d import Axes3D

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)
        from math import pi
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([1.5, 3.5])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel('J1 Base')
        self.ax.set_ylabel('J2 Shoudler')
        self.ax.set_zlabel('J3 Elbow')
        for grid in self.grids:
            grid.draw(self.ax, should_show_pca=False,
                      show_gridpoint_comps=show_gridpoint_comps)

        plt.show()

    def predict_next_pos(self, pos):
        delta = pos-self.C_space_arm_pos
        gradient = self.grid.grad_potential_energy(pos)
        print("Gradient is {}".format(gradient))
        next_position = self.C_space_arm_pos + \
            delta-gradient*(delta/self.SPEED)**2/2
        print("Next position is {}".format(next_position))
        for i in range(3):
            self.C_space_arm_pos[i] = next_position[i]
        return next_position

    def train(self, csv_file, iteration=3, show_plot=False, show_samples=True, k=1):

        if csv_file:
            self.visual = DataVisual(
                csvfile=csv_file, C_space_arm_pos=self.C_space_arm_pos, k=k)

        self.grids = self.visual.grids
        if iteration > 0:
            for i in range(iteration):
                # training
                for j in range(k):
                    self.grids[j].update_gridpoints()

        if show_plot:
            # print("run here")
            # plt.ion()
            self.visual.scatter_plot3D(
                self.visual.C_points, draw_samples=show_samples)
            # plt.show()
            # time.sleep(1)

    def copy(self, model):
        self.grids = model.grids

    @property
    def C_space_arm_pos(self):
        return self._C_space_arm_pos

    @C_space_arm_pos.setter
    def C_space_arm_pos(self, C_space_arm_pos):
        #TODO this time the arm just using 3 actuators, 
        if C_space_arm_pos.shape == (3,): 
            self._C_space_arm_pos = C_space_arm_pos
        else:
            print("Wrong type of data")


def load_model(model_file_path):

    with open(model_file_path, "rb") as file_to_read:
        loaded_object = pickle.load(file_to_read)
    return loaded_object


def save_model(model, file_path=None):

    if not file_path:
        file_path = f'models/model{date.today()}.mdl'
    with open(file_path, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    csvfile = pathlib.Path().absolute()/"datalog_sim2022-07-26.csv"
    C_space_arm_pos = np.array([1.2, 1.3, 0.2])
    model = Model(C_space_arm_pos=C_space_arm_pos)
    model.train(csv_file=csvfile, show_plot=True)
    print("ok run in model.py")
