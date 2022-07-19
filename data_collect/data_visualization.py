import math
import numpy as np
# import sympy as sy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")
# import matplotlib.backends.backend_agg as agg
from mpl_toolkits.mplot3d import Axes3D
import pathlib
from sklearn.decomposition import PCA
# from matplotlib.mlab import PCA as mlabPCA
from time import sleep, time
from math import sqrt, exp

if __name__=="__main__":
    from grid_learning import *
else:
    from data_collect.grid_learning import *



class DataVisual():
    def __init__(self, csvfile=None, end_effector=None) -> None:
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)
        # self.canvas = agg.FigureCanvasAgg(self.fig)
        # self.canvas.draw()

        if csvfile:
            self.df = pd.read_csv(csvfile)
            self.df.set_index("timestamp", inplace=True)
            
            self.C_points = self.df[['J1_base','J2_shoulder','J3_elbow']]#.assign(Index=range(len(self.df))).set_index('Index')
            # print(self.C_points.head(3))
            self.nor_C_points = self.normalized_df(self.C_points)
            # print(self.C_points.head(3))
            self.S = self.covariance_matrix(self.nor_C_points)
            # print(self.S)
            self.grid = Grid(self.C_points)

            self.whereis = pd.DataFrame(index=self.C_points.index,columns=['Position', 'NearestGrid'])
            for i in range(len(self.C_points)):
                pos = np.array([self.C_points['J1_base'].iloc[i],self.C_points['J2_shoulder'].iloc[i],self.C_points['J3_elbow'].iloc[i]])
                self.whereis['Position'].iloc[i]=pos

            # print(self.whereis.head(5))
            
            self.find_gridpoint_of_data()

            self.set_gridpoint_around()
        
        if end_effector is not None: #end_effector should be a numpy array
            self.end_effector = end_effector
            self.ee_plot, = self.ax.plot(self.end_effector[0], self.end_effector[1], self.end_effector[2], 'ys')

    def update_end_effector(self,marker=None):
        print("End effector array {}".format(self.end_effector))
        # updating data values
        if marker:
            self.ee_plot.set_marker(marker)
        self.ee_plot.set_data_3d(self.end_effector)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_gridpoint_around(self):
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                self.grid.index(i,j,start='bottom-left').set_around_samples(self.grid.index(i,j,start='bottom-left').samples)
                if i>0 and j>0:
                    self.grid.index(i,j,start='bottom-left').set_around_samples(self.grid.index(i-1,j-1,start='bottom-left').samples)
                if i>0:
                    self.grid.index(i,j,start='bottom-left').set_around_samples(self.grid.index(i-1,j,start='bottom-left').samples)
                if j>0:
                    self.grid.index(i,j,start='bottom-left').set_around_samples(self.grid.index(i,j-1,start='bottom-left').samples)

        
    def find_gridpoint_of_data(self):
        self.grid.reset_samples()
        for index in range(len(self.whereis)):
            data_pos = self.whereis['Position'].iloc[index]
            # grid_distance = np.zeros((self.grid.size,self.grid.size))
            gridpoints = []
            for i in range(self.grid.size):
                for j in range(self.grid.size):
                    if self.grid.index(i,j,start='bottom-left').is_point_in_grid(data_pos):
                        gridpoints.append(self.grid.index(i,j,start='bottom-left'))
            
            if gridpoints:
                np_gridpoints = np.empty(len(gridpoints))
                for k, point in enumerate(gridpoints):
                    np_gridpoints[k] = np.linalg.norm(data_pos-point.pos)
                nearest_point = gridpoints[np.argmin(np_gridpoints)]
                
                        # is_there = True
                        # if i+1<self.grid.size and j+1<self.grid.size:
                        #     if self.grid.index(i+1,j+1,start='bottom-left').is_point_in_grid(data_pos):
                        #         is_there=False
                        # if i+1<self.grid.size:
                        #     if self.grid.index(i+1,j,start='bottom-left').is_point_in_grid(data_pos):
                        #         is_there=False
                        # if j+1<self.grid.size:
                        #     if self.grid.index(i,j+1,start='bottom-left').is_point_in_grid(data_pos):
                        #         is_there=False
                        # if is_there:
                self.whereis['NearestGrid'].iloc[index]=(nearest_point.i,nearest_point.j)
                nearest_point.append_to_samples(data_pos)
                #             break
                # else:
                #     continue
                # break

        # print(self.whereis.notna())

            

    def scatter_plot3D(self, data, draw_samples=True): 
        x = data['J1_base']
        y = data['J2_shoulder']
        z = data['J3_elbow']
        if draw_samples:
            self.ax.scatter(x,y,z)

        self.ax.set_xlabel('J1 Base')
        self.ax.set_ylabel('J2 Shoudler')
        self.ax.set_zlabel('J3 Elbow')

        self.ax.set_xlim([0,3])
        self.ax.set_ylim([0,3])
        self.ax.set_zlim([-2,1])

        
        self.grid.draw(ax=self.ax)
            

        # make simple, bare axis lines through space:
        # xAxisLine = ((min(x), max(x)), (0, 0), (0,0))
        # self.ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        # yAxisLine = ((0, 0), (min(y), max(y)), (0,0))
        # self.ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        # zAxisLine = ((0, 0), (0,0), (min(z), max(z)))
        # self.ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
        
        # self.make_grid(self.base_point, self.comps)
        # for grids in self.grids:
        #     self.ax.scatter(grids[:,:,0],grids[:,:,1],grids[:,:,2], s=0.8)

        # for i in range(len(self.C_points)-1):
        #     self.match2points(i, i+1)
            # self.match2points(4, 2)
        # linx, liny, linz = [x[1], x[3]], [y[1], y[3]], [z[1], z[3]]
        
        # self.ax.plot(linx, liny, linz, 'r--')
        # ax2 = fig.add_subplot(122,projection='3d')

        # # Plot a basic wireframe.
        # ax2.plot_trisurf(x, y, z, linewidth=0, antialiased=False)

        # ax2.plot_wireframe(x, y, z, rstride=10, cstride=10)


    def normalized_df(self, df):
        return (df - np.mean(df, axis=0)) / np.std(df, axis=0)

    def covariance_matrix(self, df):
        return np.dot(df.T, df) / float(df.shape[0])

    def maximize_f(self, S, x,y,z):
        return x**2 + y**2 + z**2 + (2*S[0][1]*x*y) + (2*S[0][2]*x*z) + (2*S[1][2]*y*z)

    def constrain_f(self, x,y,z):
        return x**2 + y**2 + z**2 - 1

    def norm(self, vec):
        return math.sqrt(np.dot(vec, vec))

    def match2points(self, p1, p2):
        p1 = self.C_points.iloc[p1] 
        p2 = self.C_points.iloc[p2]
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        
        self.ax.plot(linx, liny, linz, 'r--')


if __name__=="__main__":
    csvfile = pathlib.Path().absolute()/"datalog/datalog2022-06-19.csv"
    end_effector = np.array([1.2, 1.3, 0.2])
    plt.ion()
    visual = DataVisual(csvfile=csvfile, end_effector=end_effector)
    # visual.scatter_plot3D(visual.C_points,draw_samples=True)
    for n in range(10):
        print("Loop {}".format(n))
        # visual.fig.clear()
        
        # visual.fig.canvas.draw()
        # visual.fig.canvas.flush_events()
        visual.grid.update_gridpoints()
        #TODO  need to reset samples and around samples before do it
        visual.find_gridpoint_of_data()
        visual.set_gridpoint_around()
        # visual.scatter_plot3D(visual.C_points,draw_samples=False)
        # plt.pause(0.2)

    visual.scatter_plot3D(visual.C_points,draw_samples=True)
    
    # visual.grid.index(-1,1).draw_samples(visual.ax)
    
    # visual.grid.show_one_grid_point(visual.ax,2,0)
    # plt.show()
    while True:
        # p1 = input("Input end-effector position[1]: ")
        # p2 = input("Input end-effector position[2]: ")
        # p3 = input("Input end-effector position[3]: ")
        # end_effector[0] = float(p1)
        # end_effector[1] = float(p2)
        # end_effector[2] = float(p3)
        marker = input("In put marker type: ")
        visual.update_end_effector(marker=marker)
    
        
