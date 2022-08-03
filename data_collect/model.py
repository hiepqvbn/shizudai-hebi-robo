from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
if __name__=="__main__":
    from data_visualization import *
else:
    from data_collect.data_visualization import *


class Model():
    SPEED = 0.01
    def __init__(self,end_effector,forsave=False) -> None:
        self._end_effector = end_effector
        self.forsave = forsave

    def show_model(self, show_gridpoint_comps=True):
        from mpl_toolkits.mplot3d import Axes3D
        
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)
        from math import pi
        self.ax.set_xlim([0,2])
        self.ax.set_ylim([1.5,3.5])
        self.ax.set_zlim([-1,1])
        self.ax.set_xlabel('J1 Base')
        self.ax.set_ylabel('J2 Shoudler')
        self.ax.set_zlabel('J3 Elbow')
        for grid in self.grids:
            grid.draw(self.ax, should_show_pca=False, show_gridpoint_comps=show_gridpoint_comps)

        plt.show()

    def cal_next_position(self, pos):
        delta = pos-self.end_effector
        gradient = self.grid.grad_potential_energy(pos)
        print("Gradient is {}".format(gradient))
        next_position = self.end_effector+delta-gradient*(delta/self.SPEED)**2/2
        print("Next position is {}".format(next_position))
        for i in range(3):
            self.end_effector[i] = next_position[i]
        return next_position

    def train(self, csv_file, iteration=3, show_plot=False, show_samples=True, k=1):

        if csv_file:
            self.visual = DataVisual(csvfile=csv_file, end_effector=self.end_effector, k=k)

        self.grids = self.visual.grids
        if iteration>0:
            for i in range(iteration):
                #training
                for j in range(k): 
                    self.grids[j].update_gridpoints()
                
        
        if show_plot:
            # print("run here")
            # plt.ion()
            self.visual.scatter_plot3D(self.visual.C_points,draw_samples=show_samples)
            # plt.show()
            # time.sleep(1)

    def copy(self, model):
        self.grids = model.grids
        

    def save(self, model):
        import pickle
        from datetime import date
        filename = "model{}.mdl".format(date.today())
        _model = Model(end_effector=model.end_effector, forsave=True)
        _model.copy(model)
        with open(filename, 'wb') as outp:
            pickle.dump(_model, outp, pickle.HIGHEST_PROTOCOL)

    @property
    def end_effector(self):
        return self._end_effector

    @end_effector.setter
    def end_effector(self, ee):
        if ee.shape ==(3,):
            self._end_effector = ee
        else:
            print("Wrong type of data")

    @classmethod
    def load(cls, model_file):
        import pickle
        with open(model_file, "rb") as file_to_read:
            loaded_object = pickle.load(file_to_read)
        return loaded_object

if __name__=="__main__":
    csvfile = pathlib.Path().absolute()/"datalog_sim2022-07-26.csv"
    end_effector = np.array([1.2, 1.3, 0.2])
    model = Model(end_effector=end_effector)
    model.train(csv_file=csvfile, show_plot=True)
    print("ok run in model.py") 