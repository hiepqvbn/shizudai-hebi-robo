import math, time
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


N=3    #->grid size = 2N-1

class GridPoint(object):
    ALPHA1 = 0.1
    ALPHA2 = 0.1
    LAMBDA = 10**(-6)
    K=10**(-2)
    def __init__(self, i, j, C_pos, unit_vector) -> None:
        self._i = i
        self._j = j
        self._C_pos = C_pos #position of grid point in C-space np.array([theta1, theta2, theta3])
        self._es = np.copy(unit_vector)
        self._samples = []
        self._around_samples = []
        # print(self)

    def __str__(self) -> str:
        return "Gridpoint: Index[{},{}], Position [{}], Components [e1,e2,e3]:[{},{},{}]".format(self.i,self.j, self.pos, self.e1,self.e2,self.e3)

    def potential_energy(self, pos):
        h = self.project_high(pos)
        if h>self.LAMBDA:
            partialE=0
        else:
            partialE=self.K*(self.LAMBDA-h)**2/2.0

        # print("{} partial energy {}".format(self, partialE))

        return partialE

    def update_pos(self, new_pos):
        self._C_pos=new_pos

    def update_unit_vector(self, grid):
        # Update e1
        if self.i<grid.N-1:
            self._es[0]=(grid.index(self.i+1,self.j).pos-self.pos)/np.linalg.norm(grid.index(self.i+1,self.j).pos-self.pos)
        else:   #gridpoint on the edge (i=N-1)
            self._es[0]=grid.index(self.i-1,self.j).e1
        # Update e2
        if self.j<grid.N-1:
            self._es[1]=(grid.index(self.i,self.j+1).pos-self.pos)/np.linalg.norm(grid.index(self.i,self.j+1).pos-self.pos)
        else:
            self._es[1]=grid.index(self.i,self.j-1).e2
        # Update e3
        self._es[2] = -np.cross(self.e1,self.e2)


    def project_high(self, pos):
        """
        e3 vector direction height (positive/negative)
        """
        return np.dot(self.e3,(pos-self.pos))
    
    def point_weight_e1(self, pos):
        """
        weight function of sample point p with this Gridpoint
        """
        p=pos-self.pos
        weight = np.abs(np.dot(p,self.e1))

        weight = math.exp(-weight**2/self.ALPHA1**2)
        if np.isnan(weight):
            print("pos is {}".format(self.pos))
            print("e1 is {}".format(self.e1))
            print("weight is {}".format(weight))
        return weight

    def point_weight_e2(self, pos):
        """
        weight function of sample point p with this Gridpoint
        """
        p=pos-self.pos
        weight = np.abs(np.dot(p,self.e2))

        weight = math.exp(-weight**2/self.ALPHA2**2)
        if np.isnan(weight):
            print("pos is {}".format(self.pos))
            print("e1 is {}".format(self.e1))
            print("weight is {}".format(weight))
        return weight

    def is_point_in_grid(self, pos):
        if np.dot(self.e1,(pos-self.pos))<0: 
            return False
        elif np.dot(self.e2,(pos-self.pos))<0:
            return False
        else:
            return True
    
    def draw_samples(self, fig):
        np_samples = np.array(self._samples)
        # print(np_samples)
        if np_samples.size != 0:
            fig.plot(np_samples[:,0],np_samples[:,1],np_samples[:,2], 'yo')

    def reset_samples(self):
        self._samples = []
        self._around_samples = []

    def draw_comps(self, fig):
        """
        Draw e1, e2, e3 vector of Gridpoint
        """
        rate=0.12
        for ii, comp in enumerate(self.components):
            # if not ii==2:    #dont show e3     
            fig.plot(
                [self.pos[0], self.pos[0]+comp[0]*rate],
                [self.pos[1], self.pos[1]+comp[1]*rate],
                [self.pos[2], self.pos[2]+comp[2]*rate],
                label=f"Component {ii}",
                linewidth=1,
                color=f"C{ii + 2}",
            )

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
    def components(self):
        return self._es

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
    def __init__(self, data, basepoint=None) -> None:
        # self.data = data[data['Kmeans_label']==0]
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
       
        self.width_1=0.8*(np.dot(self.np_samples,self.comps[0]).max()-np.dot(self.np_samples,self.comps[0]).min())/(self.N*2-2)
        self.width_2=0.8*(np.dot(self.np_samples,self.comps[1]).max()-np.dot(self.np_samples,self.comps[1]).min())/(self.N*2-2)

        self.make_init_grid()
        self.make_numpy_grid()

        self.make_point_list()

        self.whereis = pd.DataFrame(index=self.data.index,columns=['Position', 'NearestGrid'])

        for i in range(len(self.data)):
            pos = np.array([self.data['J1_base'].iloc[i],self.data['J2_shoulder'].iloc[i],self.data['J3_elbow'].iloc[i]])
            self.whereis['Position'].iloc[i]=pos

        # print(self.whereis.head(5))
        
        self.find_gridpoint_of_data()

        self.set_gridpoint_around()
        
    def set_gridpoint_around(self):
        for i in range(self.size):
            for j in range(self.size):
                self.index(i,j,start='bottom-left').set_around_samples(self.index(i,j,start='bottom-left').samples)
                if i>0 and j>0:
                    self.index(i,j,start='bottom-left').set_around_samples(self.index(i-1,j-1,start='bottom-left').samples)
                if i>0:
                    self.index(i,j,start='bottom-left').set_around_samples(self.index(i-1,j,start='bottom-left').samples)
                if j>0:
                    self.index(i,j,start='bottom-left').set_around_samples(self.index(i,j-1,start='bottom-left').samples)

        
    def find_gridpoint_of_data(self):
        self.reset_samples()
        for index in range(len(self.whereis)):
            data_pos = self.whereis['Position'].iloc[index]
            # grid_distance = np.zeros((self.grid.size,self.grid.size))
            gridpoints = []
            for i in range(self.size):
                for j in range(self.size):
                    if self.index(i,j,start='bottom-left').is_point_in_grid(data_pos):
                        gridpoints.append(self.index(i,j,start='bottom-left'))
            
            if gridpoints:
                np_gridpoints = np.empty(len(gridpoints))
                for k, point in enumerate(gridpoints):
                    np_gridpoints[k] = np.linalg.norm(data_pos-point.pos)
                nearest_point = gridpoints[np.argmin(np_gridpoints)]
                
                self.whereis['NearestGrid'].iloc[index]=(nearest_point.i,nearest_point.j)
                nearest_point.append_to_samples(data_pos)
                

    #Calculate potential energy of end effector(from end effector position)
    def potential_energy(self, pos):
        """
        Calculate potential energy of end effector(from end effector position\n
        Equal to sum of all gridpoint Potential Energy
        """
        E = 0.0
        for point in self.point_list:
            E += point.potential_energy(pos)
        
        return E

    def grad_potential_energy(self, pos):
        DELTA = 10**(-2)
        energy = self.potential_energy(pos)
        print("Potential energy is {}".format(energy))
        gradient = np.zeros(3)
        for i in range(3):
            temp_pos = np.copy(pos)
            temp_pos[i] = pos[i]+DELTA
            temp_energy = self.potential_energy(temp_pos)
            gradient[i] = (temp_energy-energy)/DELTA
        
        return gradient

    def draw(self, ax, should_show_center_point=True, should_show_pca=True, show_grid=True, show_gridpoint_comps=True):
        
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
        """
        # for i in range(self.N):
        #     for j in range(self.N):
        #         self.fig.scatter(self.index(i,j).pos[0],self.index(i,j).pos[1],self.index(i,j).pos[2], s=0.8)
        #         self.fig.scatter(self.index(-i,j).pos[0],self.index(-i,j).pos[1],self.index(-i,j).pos[2], s=0.8)
        #         self.fig.scatter(self.index(-i,-j).pos[0],self.index(-i,-j).pos[1],self.index(-i,-j).pos[2], s=0.8)
        #         self.fig.scatter(self.index(i,-j).pos[0],self.index(i,-j).pos[1],self.index(i,-j).pos[2], s=0.8)

        #         if t:
        #             plt.pause(t)
        """
        if show_grid:
            self.make_numpy_grid()
            ax.scatter(self.np_grid[:,:,0],self.np_grid[:,:,1],self.np_grid[:,:,2], s=9)

        if show_gridpoint_comps:
            for i in range(self.size):
                for j in range(self.size):
                    point = self.index(i,j,start='bottom-left')
                    point.draw_comps(ax)


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

    def update_gridpoints(self):
        self.update_gridpoints_e1()
        self.update_gridpoints_e2()
        self.update_gridpoints_dis()    #update distance between grid point same as weight1, weight2
        self.find_gridpoint_of_data()
        self.set_gridpoint_around()

    def update_gridpoints_dis(self):
        for point in self.point_list:
            if point.i<0:
                next_point_i = self.index(point.i+1,point.j)
                new_pos = next_point_i.pos - self.width_1*point.e1
            elif point.i>0:
                prev_point_i = self.index(point.i-1,point.j)
                new_pos = prev_point_i.pos + self.width_1*prev_point_i.e1
            elif point.i==0:
                new_pos=point.pos
            point.update_pos(new_pos)
        for point in self.point_list:
            point.update_unit_vector(self)
        for point in self.point_list:
            if point.j<0:
                next_point_j = self.index(point.i,point.j+1)
                new_pos = next_point_j.pos - self.width_2*point.e2
            elif point.j>0:
                prev_point_j = self.index(point.i,point.j-1)
                new_pos = prev_point_j.pos + self.width_2*prev_point_j.e2
            elif point.j==0:
                new_pos=point.pos
            point.update_pos(new_pos)
        for point in self.point_list:
            point.update_unit_vector(self)

    def update_gridpoints_e1(self):  
        for point in self.point_list:
            around_samples = point.around_samples
            heights = np.zeros(len(around_samples))
            if around_samples:
                for sample in range(len(around_samples)):
                    weight = point.point_weight_e1(around_samples[sample])
                    heights[sample] = weight*point.project_high(around_samples[sample])
                delta_z = point.e3*np.mean(heights)
                point.update_pos(point.pos+delta_z)
        for point in self.point_list:
            point.update_unit_vector(self)

    def update_gridpoints_e2(self):
        for point in self.point_list:
            around_samples = point.around_samples
            heights = np.zeros(len(around_samples))
            if around_samples:
                for sample in range(len(around_samples)):
                    weight = point.point_weight_e2(around_samples[sample])
                    heights[sample] = weight*point.project_high(around_samples[sample])
                delta_z = point.e3*np.mean(heights)
                point.update_pos(point.pos+delta_z)
        for point in self.point_list:
            point.update_unit_vector(self)

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

    def make_point_list(self):
        self._point_list = []
        for i in range(self.size):
            for j in range(self.size):
                point = self.index(i,j,start='bottom-left')
                self._point_list.append(point)

    @property
    def point_list(self):
        return self._point_list

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

    def reset_samples(self):
        for i in range(self.size):
            for j in range(self.size):
                self.index(i,j,start='bottom-left').reset_samples()

    @property
    def size(self):
        return self.N*2-1

