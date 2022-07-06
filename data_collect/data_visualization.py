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
from time import sleep, time

N=10    #->grid size = 2N-1

class GridPoint(object):
    def __init__(self, i, j, C_pos, unit_vector) -> None:
        self._i = i
        self._j = j
        self._C_pos = C_pos #position of grid point in C-space np.array([theta1, theta2, theta3])
        self._es = unit_vector
        self._samples = []
        self._around_samples = []

    def project_high(self, pos):
        return np.abs(np.dot(self.e3,(pos-self.pos)))

    def is_point_in_grid(self, pos):
        if np.dot(self.e1,(pos-self.pos))<0: 
            return False
        elif np.dot(self.e2,(pos-self.pos))<0:
            return False
        else:
            return True
    
    def draw_samples(self, fig):
        np_samples = np.array(self._samples)
        print(np_samples)
        if np_samples.size != 0:
            fig.plot(np_samples[:,0],np_samples[:,1],np_samples[:,2], 'yo')

    @property
    def around_samples(self):
        return self._around_samples

    def set_around_samples(self, samples):
        self._around_samples.extend(samples)

    @property
    def samples(self):
        return self._samples

    def append_to_samples(self, point):
        self._samples.append(point)

    def reset_samples(self):
        self._samples = []

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
    def pos(self):
        return self._C_pos

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
    def __init__(self, data, basepoint=None, N=10) -> None:
        self.data = data
        
        self.N = N
        if basepoint:
            self.basepoint = basepoint
        else:
            self.basepoint = self.get_center(data)

        self.pca = PCA(n_components=len(self.data.columns))
        self.pca.fit(self.data)
        self.pca_result=pd.DataFrame(self.pca.transform(data), columns=data.columns)
        self.comps = self.pca.components_
        self.vars = self.pca.explained_variance_

        self.np_samples = self.data.to_numpy()
       
        self.width_1=1.2*(np.dot(self.np_samples,self.comps[0]).max()-np.dot(self.np_samples,self.comps[0]).min())/(self.N*2-2)
        self.width_2=1.2*(np.dot(self.np_samples,self.comps[1]).max()-np.dot(self.np_samples,self.comps[1]).min())/(self.N*2-2)

        self.make_init_grid()
        self.make_numpy_grid()


    def draw(self, ax, should_show_center_point=True, should_show_pca=True, show_grid=True):
        
        if should_show_center_point:
            ax.plot(self.basepoint[0],self.basepoint[1],self.basepoint[2], 'rx')
        if should_show_pca:
            for i, (comp, var) in enumerate(zip(self.pca.components_, self.pca.explained_variance_)):
        
                comp = comp * var  # scale component by its variance explanation power
        
                ax.plot(
                    [self.basepoint[0], self.basepoint[0]+comp[0]*(i**5+5)],
                    [self.basepoint[1], self.basepoint[1]+comp[1]*(i**5+5)],
                    [self.basepoint[2], self.basepoint[2]+comp[2]*(i**5+5)],
                    label=f"Component {i}",
                    linewidth=2,
                    color=f"C{i + 2}",
                )
        # for i in range(self.N):
        #     for j in range(self.N):
        #         self.fig.scatter(self.index(i,j).pos[0],self.index(i,j).pos[1],self.index(i,j).pos[2], s=0.8)
        #         self.fig.scatter(self.index(-i,j).pos[0],self.index(-i,j).pos[1],self.index(-i,j).pos[2], s=0.8)
        #         self.fig.scatter(self.index(-i,-j).pos[0],self.index(-i,-j).pos[1],self.index(-i,-j).pos[2], s=0.8)
        #         self.fig.scatter(self.index(i,-j).pos[0],self.index(i,-j).pos[1],self.index(i,-j).pos[2], s=0.8)

        #         if t:
        #             plt.pause(t)
        if show_grid:
            ax.scatter(self.np_grid[:,:,0],self.np_grid[:,:,1],self.np_grid[:,:,2], s=0.8)

    def show_one_grid_point(self, fig, i, j):
        gridpoints = self.find_grid_around_gridpoint(i,j)
        np_grids=np.empty((len(gridpoints),3))
        gridpoints[0].draw_samples(fig)
        for p in range(len(gridpoints)):
            np_grids[p]=gridpoints[p].pos
            
        fig.plot(np_grids[0,0],np_grids[0,1],np_grids[0,2], 'go')
        fig.plot(np_grids[1:,0],np_grids[1:,1],np_grids[1:,2], 'ro')
        np_samples = np.array(self.index(i,j).around_samples)
        if np_samples.size != 0:
            fig.plot(np_samples[:,0],np_samples[:,1],np_samples[:,2], 'bx')

    def find_grid_around_gridpoint(self, i, j, start='center'):
        gridpoints=[]
        if start=='center':
            gridpoints.append(self.index(i,j))
            try:
                gridpoints.append(self.index(i+1,j+1))
            except: pass
            try:
                gridpoints.append(self.index(i,j+1))
            except: pass
            try:
                gridpoints.append(self.index(i-1,j+1))
            except: pass
            try:
                gridpoints.append(self.index(i-1,j))
            except: pass
            try:
                gridpoints.append(self.index(i-1,j-1))
            except: pass
            try:
                gridpoints.append(self.index(i,j-1))
            except: pass
            try:
                gridpoints.append(self.index(i+1,j-1))
            except: pass
            try:
                gridpoints.append(self.index(i+1,j))
            except: pass
        return gridpoints

    def index(self, i, j, start='center'):  #grid point at index i,j <phi(i,j)> with i, j belong to Z (integer)
        """
        default: start='center'\n
        switch to start='bottom-left' to start index from bottom left
        """
        if start=='center':
            if j<0:
                return self.grid[2][-i][-j] if i<0 else self.grid[3][i][-j] # third descartes quadrant or forth descartes quadrant
            else:
                return self.grid[1][-i][j] if i<0 else self.grid[0][i][j] # second descartes quadrant or first descartes quadrant
        elif start=='bottom-left':
            i=i+1-self.N
            j=j+1-self.N
            return self.index(i,j)


    def make_numpy_grid(self):
        self.np_grid = np.empty((self.N*2-1,self.N*2-1,3))
        for i in range(self.N*2 - 1):
            for j in range(self.N*2-1):
                self.np_grid[i,j]=self.index(i+1-self.N, j+1-self.N).pos

    def make_init_grid(self):
        self.grid = [[],[],[],[]]
        
        for i in range(self.N):
            for grid in self.grid:
                grid.append([])
            for j in range(self.N):
                self.grid[0][i].append(GridPoint(i,j,self.get_C_pos(i,j),self.comps))
                self.grid[1][i].append(GridPoint(-i,j,self.get_C_pos(-i,j),self.comps))
                self.grid[2][i].append(GridPoint(-i,-j,self.get_C_pos(-i,-j),self.comps))
                self.grid[3][i].append(GridPoint(i,-j,self.get_C_pos(i,-j),self.comps))
        
    def get_C_pos(self,i,j):
        if (i,j)==(0,0):
            c_pos = self.basepoint
        else:
            if j==0:
                prev_i_point = self.index(i-1,0) if i>0 else self.index(i+1,0) #if i<0
                c_pos = prev_i_point.pos + self.width_1*prev_i_point.e1 if i>0 else prev_i_point.pos - self.width_1*prev_i_point.e1
            else:
                prev_j_point = self.index(i,j-1) if j>0 else self.index(i,j+1) #if j<0
                c_pos = prev_j_point.pos + self.width_2*prev_j_point.e2 if j>0 else prev_j_point.pos - self.width_2*prev_j_point.e2
        return c_pos

    def get_center(self, data):
        x = data['J1_base']
        y = data['J2_shoulder']
        z = data['J3_elbow']
        return np.array([np.mean(x), np.mean(y), np.mean(z)])

    @property
    def size(self):
        return self.N*2-1



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
            self.grid = Grid(self.C_points,N=7)

            self.whereis = pd.DataFrame(index=self.C_points.index,columns=['Position', 'NearestGrid'])
            for i in range(len(self.C_points)):
                pos = np.array([self.C_points['J1_base'].iloc[i],self.C_points['J2_shoulder'].iloc[i],self.C_points['J3_elbow'].iloc[i]])
                self.whereis['Position'].iloc[i]=pos

            # print(self.whereis.head(5))
            
            self.find_gridpoint_of_data()

            self.set_gridpoint_around()

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
        for index in range(len(self.whereis)):
            data_pos = self.whereis['Position'].iloc[index]
            # grid_distance = np.zeros((self.grid.size,self.grid.size))
            for i in range(self.grid.size):
                for j in range(self.grid.size):
                    if self.grid.index(i,j,start='bottom-left').is_point_in_grid(data_pos):
                        is_there = True
                        if i+1<self.grid.size and j+1<self.grid.size:
                            if self.grid.index(i+1,j+1,start='bottom-left').is_point_in_grid(data_pos):
                                is_there=False
                        if i+1<self.grid.size:
                            if self.grid.index(i+1,j,start='bottom-left').is_point_in_grid(data_pos):
                                is_there=False
                        if j+1<self.grid.size:
                            if self.grid.index(i,j+1,start='bottom-left').is_point_in_grid(data_pos):
                                is_there=False
                        if is_there:
                            self.whereis['NearestGrid'].iloc[index]=(self.grid.index(i,j,start='bottom-left').i,self.grid.index(i,j,start='bottom-left').j)
                            self.grid.index(i,j,start='bottom-left').append_to_samples(data_pos)
                            break
                else:
                    continue
                break

        print(self.whereis.notna())

            

    def scatter_plot3D(self, data, draw_samples=True):
        # pca = PCA(n_components=3)
        # pca.fit(data)
        # result=pd.DataFrame(pca.transform(data), columns=data.columns)
        
        

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)
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


        # if should_show_center_point:
        #     self.base_point = np.array([np.mean(x), np.mean(y), np.mean(z)])
        #     # print(self.base_point)
            

        # make simple, bare axis lines through space:
        # xAxisLine = ((min(x), max(x)), (0, 0), (0,0))
        # self.ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        # yAxisLine = ((0, 0), (min(y), max(y)), (0,0))
        # self.ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        # zAxisLine = ((0, 0), (0,0), (min(z), max(z)))
        # self.ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
        
        # self.comps = []
        
        
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
    visual.scatter_plot3D(visual.C_points,draw_samples=False)
    # visual.grid.index(-1,1).draw_samples(visual.ax)
    visual.grid.show_one_grid_point(visual.ax,2,0)
    plt.show()
    
        
