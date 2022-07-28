from data_collect.model import Model
import pathlib
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    csvfile = pathlib.Path().absolute()/"datalog/datalog_sim2022-07-27.csv"
    end_effector = np.array([1.2, 1.3, 0.2])
    # plt.ion()

    model = Model.load("model2022-07-28.mdl")
    if model:
        # print(model.grids[0].index(1,2))
        
        # e1,e2, e3 = model.grids[0].index(1,2).e1,model.grids[0].index(1,2).e2,model.grids[0].index(1,2).e3
        # print(e2)
        # e2 = np.cross(e1,e3)
        
        # print(e2)
        # print(model.grids[0].index(1,2).components)
        # print(np.dot(model.grids[0].index(1,2).e2,model.grids[0].index(1,2).e3))
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = Axes3D(fig)
        from math import pi
        ax.set_xlim([0,2])
        ax.set_ylim([1.5,3.5])
        ax.set_zlim([-1,1])
        ax.set_xlabel('J1 Base')
        ax.set_ylabel('J2 Shoudler')
        ax.set_zlabel('J3 Elbow')
        model.grids[0].draw(ax, should_show_pca=False, show_gridpoint_comps=True)

        plt.show()


    # model = Model(end_effector=end_effector)
    # model.train(csv_file=csvfile, iteration=6, show_plot=True, show_samples=True)
    # model.save(model)
    # print("olk")
    # plt.show()



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