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
    def __init__(self,end_effector) -> None:
        self.end_effector = end_effector

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
            self.visual = DataVisual(csvfile=csv_file, end_effector=self.end_effector)

        self.grids = self.visual.grids

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

        

    def save(self):
        import pickle
        from datetime import date
        filename = "model{}".format(date.today())
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, model_file):
        pass

if __name__=="__main__":
    csvfile = pathlib.Path().absolute()/"datalog_sim2022-07-26.csv"
    end_effector = np.array([1.2, 1.3, 0.2])
    model = Model(end_effector=end_effector)
    model.train(csv_file=csvfile, show_plot=True)
    print("ok run in model.py") 