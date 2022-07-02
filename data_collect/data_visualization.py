import math
import numpy as np
from pyparsing import col
import sympy as sy
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib
from sklearn.decomposition import PCA
# from matplotlib.mlab import PCA as mlabPCA


class GridPoint(object):
    def __init__(self, i, j, C_pos, unit_vector) -> None:
        self._i = i
        self._j = j
        self._C_pos = C_pos #position of grid point in C-space np.array([theta1, theta2, theta3])
        self._es = unit_vector

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @property
    def index(self):
        return (self.i, self.j)

    @property
    def e1(self):
        return self._es[0]

    @property
    def e2(self):
        return self._es[1]

    @property
    def e3(self):
        return self._es[2]


class Grid(object):
    def __init__(self, data, basepoint=None, width=0.2, N=10) -> None:
        self.data = data
        self.N = N
        self.width =  width
        if basepoint:
            self.basepoint = basepoint
        else:
            self.basepoint = self.get_center(data)

        self.pca = PCA(n_components=len(self.data.columns))
        self.pca.fit(self.data)
        self.pca_result=pd.DataFrame(self.pca.transform(data), columns=data.columns)
        self.comps = self.pca.components_

        self.make_init_grids()

    def index(self, i, j):  #grid point at index i,j <phi(i,j)> with i, j belong to Z (integer)
        if j<0:
            return self.grid[2][-i][-j] if i<0 else self.grid[3][i][-j] # third descartes quadrant or forth descartes quadrant
        else:
            return self.grid[1][-i][j] if i<0 else self.grid[1][i][j] # second descartes quadrant or first descartes quadrant

    def make_init_grid(self):
        self.grid = [[],[],[],[]]
        for grid in self.grid:
            grid.append(GridPoint(0,0,self.basepoint,self.comps))
        for i in range(1,self.N):
            for j in range(1,self.N):
                self.grid[0].append(GridPoint(i,j,self.get_C_pos(i,j),self.comps))
                self.grid[1].append(GridPoint(-i,j,self.get_C_pos(-i,j),self.comps))
                self.grid[2].append(GridPoint(-i,-j,self.get_C_pos(-i,-j),self.comps))
                self.grid[3].append(GridPoint(i,-j,self.get_C_pos(i,-j),self.comps))
        
        for k in range(4):
            for i in range(len(self.grids[k])):
                self.grid[k][i][0] = self.basepoint + i*self.comps[0]*self.width if k%2==0 else self.basepoint - i*self.comps[0]*self.width
                for j in range(len(self.grids[k][i])):
                    self.grid[k][i][j] = self.grids[k][i][0] + j*self.comps[1]*self.width if k<2 else self.grids[k][i][0] - j*self.comps[1]*self.width
        
    def get_C_pos(self,i,j):
        prev_i_point = self.index()
        prev_j_point = self.index()
        c_pos = 0
        return c_pos

    def get_center(self, data):
        x = data['J1_base']
        y = data['J2_shoulder']
        z = data['J3_elbow']
        return np.array([np.mean(x), np.mean(y), np.mean(z)])



class DataVisual():
    def __init__(self, csvfile=None) -> None:
        if csvfile:
            self.df = pd.read_csv(csvfile)
            self.df.set_index("timestamp", inplace=True)
            
            self.C_points = self.df[['J1_base','J2_shoulder','J3_elbow']]#.assign(Index=range(len(self.df))).set_index('Index')
            # print(self.C_points.head(3))
            self.nor_C_points = self.normalized_df(self.C_points)
            # print(self.C_points.head(3))
            self.S = self.covariance_matrix(self.nor_C_points)
            # print(self.S)

            

    def scatter_plot3D(self, data, should_show_center_point=True):
        pca = PCA(n_components=3)
        pca.fit(data)
        result=pd.DataFrame(pca.transform(data), columns=data.columns)
        


        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)
        x = data['J1_base']
        y = data['J2_shoulder']
        z = data['J3_elbow']
        self.ax.scatter(x,y,z)
        self.ax.set_xlabel('J1 Base')
        self.ax.set_ylabel('J2 Shoudler')
        self.ax.set_zlabel('J3 Elbow')

        self.ax.set_xlim([0,3])
        self.ax.set_ylim([0,3])
        self.ax.set_zlim([-2,1])


        if should_show_center_point:
            self.base_point = np.array([np.mean(x), np.mean(y), np.mean(z)])
            # print(self.base_point)
            self.ax.plot([np.mean(x)],[np.mean(y)],[np.mean(z)], 'rx')

        # make simple, bare axis lines through space:
        # xAxisLine = ((min(x), max(x)), (0, 0), (0,0))
        # self.ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        # yAxisLine = ((0, 0), (min(y), max(y)), (0,0))
        # self.ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        # zAxisLine = ((0, 0), (0,0), (min(z), max(z)))
        # self.ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
        
        self.comps = []
        for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
            # print(np.linalg.norm(comp))
            self.comps.append(comp)
            comp = comp * var  # scale component by its variance explanation power
            
            self.ax.plot(
                [np.mean(x), np.mean(x)+comp[0]*(i**5+5)],
                [np.mean(y), np.mean(y)+comp[1]*(i**5+5)],
                [np.mean(z), np.mean(z)+comp[2]*(i**5+5)],
                label=f"Component {i}",
                linewidth=2,
                color=f"C{i + 2}",
            )
        
        self.make_grid(self.base_point, self.comps)
        for grids in self.grids:
            self.ax.scatter(grids[:,:,0],grids[:,:,1],grids[:,:,2], s=0.8)

        # for i in range(len(self.C_points)-1):
        #     self.match2points(i, i+1)
            # self.match2points(4, 2)
        # linx, liny, linz = [x[1], x[3]], [y[1], y[3]], [z[1], z[3]]
        
        # self.ax.plot(linx, liny, linz, 'r--')
        # ax2 = fig.add_subplot(122,projection='3d')

        # # Plot a basic wireframe.
        # ax2.plot_trisurf(x, y, z, linewidth=0, antialiased=False)

        # ax2.plot_wireframe(x, y, z, rstride=10, cstride=10)

        

    def make_grid(self, base_point, comps, width=0.1, N=5):
        self.grids = np.zeros((4,N,N,3))
        for k in range(4):
            for i in range(len(self.grids[k])):
                self.grids[k][i][0] = base_point + i*comps[0]*width if k%2==0 else base_point - i*comps[0]*width
                for j in range(len(self.grids[k][i])):
                    self.grids[k][i][j] = self.grids[k][i][0] + j*comps[1]*width if k<2 else self.grids[k][i][0] - j*comps[1]*width
        # print("this:{}".format(self.grids))

    def grids_pos(self, i, j): #grid postition phi(i,j) with i, j belong to Z (integer)
        if j<0:
            return self.grids[2][-i][-j] if i<0 else self.grids[3][i][-j] # third descartes quadrant or forth descartes quadrant
        else:
            return self.grids[1][-i][j] if i<0 else self.grids[1][i][j] # second descartes quadrant or first descartes quadrant

    def normalized_df(self, df):
        return (df - np.mean(df, axis=0)) / np.std(df, axis=0)

    def covariance_matrix(self, df):
        return np.dot(df.T, df) / float(df.shape[0])

    def maximize_f(self, S, x,y,z):
        return x**2 + y**2 + z**2 + (2*S[0][1]*x*y) + (2*S[0][2]*x*z) + (2*S[1][2]*y*z)

    def constrain_f(self, x,y,z):
        return x**2 + y**2 + z**2 - 1

    def value_without_infinitesimal(sympy_value):
        if type(sympy_value) == sy.numbers.Add:
            return sympy_value.args[0]
        else:
            return sympy_value

    def norm(self, vec):
        return math.sqrt(np.dot(vec, vec))

    def lagrange_multipler(self):
        x, y, z, l = sy.symbols('x y z lamda')
        self.f = self.maximize_f(self.S, x, y, z)
        # print(self.f)
        self.g = self.constrain_f(x, y, z)
        self.L =  self.f - l*self.g

        dx = self.L.diff(x)
        dy = self.L.diff(y)
        dz = self.L.diff(z)
        dl = self.L.diff(l)*-1
        print(dx)
        print(dy)
        print(dz)
        print('here')
        extreme_value_candidates = sy.solve([dx, dy, dz, dl])
        print("hhhere")
        P = []

        for evc in extreme_value_candidates:
            print(evc)
            p = {
                x: self.value_without_infinitesimal(evc[x]),
                y: self.value_without_infinitesimal(evc[y]),
                z: self.value_without_infinitesimal(evc[z]),
                l: self.value_without_infinitesimal(evc[l])
            }
            P.append(p)
            

    def match2points(self, p1, p2):
        p1 = self.C_points.iloc[p1] 
        p2 = self.C_points.iloc[p2]
        linx, liny, linz = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
        
        self.ax.plot(linx, liny, linz, 'r--')


if __name__=="__main__":
    csvfile = pathlib.Path().absolute()/"datalog/datalog2022-06-19.csv"
    visual = DataVisual(csvfile=csvfile)
    # visual.lagrange_multipler()
    visual.scatter_plot3D(visual.C_points)
    plt.show()
    
        
