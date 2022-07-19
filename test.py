from data_collect.model import Model
import pathlib
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    csvfile = pathlib.Path().absolute()/"datalog/datalog2022-06-19.csv"
    end_effector = np.array([1.2, 1.3, 0.2])
    # plt.ion()
    model = Model(end_effector=end_effector)
    if model:
        print(model)
    # model.train(csv_file=csvfile, show_plot=True)
    print("olk")
    # new_pos = np.zeros(3)
    # while True:
    #     p1 = input("Input end-effector position[1]: ")
    #     p2 = input("Input end-effector position[2]: ")
    #     p3 = input("Input end-effector position[3]: ")
        
    #     new_pos[0] = float(p1)
    #     new_pos[1] = float(p2)
    #     new_pos[2] = float(p3)
    #     model.cal_next_position(new_pos)
    #     # print(model.visual.end_effector)
    #     model.visual.update_end_effector()