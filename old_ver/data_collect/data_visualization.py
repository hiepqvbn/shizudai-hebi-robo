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
from sklearn.cluster import KMeans
# from matplotlib.mlab import PCA as mlabPCA
from time import sleep, time
from math import sqrt, exp, pi

if __name__=="__main__":
    from grid_learning import *
else:
    try:
        from data_collect.grid_learning import *
    except:
        from grid_learning import *


class DataVisual():
    def __init__(self, csvfile=None, end_effector=None, k=1) -> None:
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)
        # self.canvas = agg.FigureCanvasAgg(self.fig)
        # self.canvas.draw()

        if csvfile:
            self.df = pd.read_csv(csvfile)
            self.df.set_index("timestamp", inplace=True)
            
            self.C_points = self.df[['J1_base','J2_shoulder','J3_elbow']]#.assign(Index=range(len(self.df))).set_index('Index')
            
            # K-means Clustering Algorithm
            # k=1 #clusters number
            self.clustering(self.C_points, k)

            # Add k-means clustering label to Configuration Points Dataframe
            self.C_points['Kmeans_label'] = self.kmeans.labels_
            # print(self.C_points.head(3))
            
            self.grids = []
            for i in range(k):
                data = self.C_points[self.C_points['Kmeans_label']==i].iloc[:,0:3]
                # print(data.head(3))
                # print("C points len:{}".format(len(self.C_points))) 
                # print("data len:{}".format(len(data)))            
                grid = Grid(data)
                # if i==0:
                #     data.to_csv("thetas.csv")
            
                self.grids.append(grid)
        
        if end_effector is not None: #end_effector should be a numpy array
            self.end_effector = end_effector
            self.ee_plot, = self.ax.plot(self.end_effector[0], self.end_effector[1], self.end_effector[2], 'ys')

    def clustering(self, data, k):
        self.kmeans = KMeans(n_clusters=k)
        self.kmeans.fit(data)
        # print("i=0 kmeans data number {}".format(np.count_nonzero(self.kmeans.labels_==0)))

    def update_end_effector(self,marker=None):
        print("End effector array {}".format(self.end_effector))
        # updating data values
        if marker:
            self.ee_plot.set_marker(marker)
        self.ee_plot.set_data_3d(self.end_effector)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            

    def scatter_plot3D(self, data, draw_samples=True, draw_grids=False, k=1): 
        x = data['J1_base']
        y = data['J2_shoulder']
        z = data['J3_elbow']
        if draw_samples:
            self.ax.scatter(x,y,z, s=0.2, c=self.kmeans.labels_, cmap='rainbow')

        self.ax.set_xlabel('J1 Base')
        self.ax.set_ylabel('J2 Shoudler')
        self.ax.set_zlabel('J3 Elbow')

        self.ax.set_xlim([0,2])
        self.ax.set_ylim([1.5,3.5])
        self.ax.set_zlim([-1,1])

        if draw_grids:
            for j in range(k):
                self.grids[j].draw(ax=self.ax)
            

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
    
        
