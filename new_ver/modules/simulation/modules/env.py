#TODO: add obstacles, for simple, just have 1 arm and 1 cam in enviroment
class Environment:
    def __init__(self, cam, arm):
        self.cam = cam
        self.arm = arm

    def visualize(self, plt):
        from mpl_toolkits.mplot3d import Axes3D

        self.plt = plt

        self.fig = self.plt.figure(figsize=(12, 9))
        self.ax = Axes3D(self.fig)

        self.fig.suptitle("3DoF Arm")

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        print(f'{self.arm.base=}')
        self.ax.set_xlim([-1, self.arm.base[0]+2])
        self.ax.set_ylim([-1, self.arm.base[1]+2])
        self.ax.set_zlim([-1, self.arm.base[2]+2])

        # Call visualize() method for each object in the environment
        self.arm.visualize(self.ax)
        self.cam.visualize_in_env(self.ax)

        # Additional visualization operations for the environment
        # ...


    def update_visualization(self):
        self.arm.update_draw()
        
        #TODO: update cam in env. for simple, cam position is fixed in (0,0,0)
        # self.cam.update_draw_in_env


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()