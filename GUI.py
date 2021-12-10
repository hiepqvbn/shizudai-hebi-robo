import tkinter as tk
import hebi
from time import sleep
import numpy as np

class Robot:
    fq_hz=0.5 #Hz
    fq = fq_hz*2*np.pi
    inertia = 0.1 #kg.m^2
    def __init__(self):
        self.connect()

    def connect(self):
        self.lookup = hebi.Lookup()
        sleep(2)
        print('connecting')
        if self.isConnected():
            self.family_names = ['Arm']#self.familyName()
            self.module_names = ['J3_elbow', 'J1_base', 'J2_shoulder', 'gripperSpool']#self.moduleName()
            print(self.family_names)
            print(self.module_names)
            self.group = self.lookup.get_group_from_names(self.family_names, self.module_names)
            self.group.feedback_frequency = 24
        else:
            self.module_names = ['a', 'b', 'c']

    def isConnected(self):
        entries = []
        for entry in self.lookup.entrylist:
            entries.append(entry)
        if entries: return True
        else: return False

    def familyName(self):
        family_names = []
        for entry in self.lookup.entrylist:
            family_name = str(entry).partition("Family:")[2].split()[0]
            if not family_name in family_names:
                family_names.append(family_name)
        return family_names

    def moduleName(self):
        module_names = []
        for entry in self.lookup.entrylist:
            module_name = str(entry).partition("Name:")[2].split()[0]
            if not module_name in module_names:
                module_names.append(module_name)
        return module_names


class App:

    def __init__(self, title):
        self.window = tk.Tk()
        self.window.title(title)
        self.robot = Robot()
        self.stt()


    def add_sliders(self):
        feedback = self.robot.group.get_next_feedback()
        x = self.robot.module_names.index(self.tkvar.get())
        
        self.pos_slider = tk.Scale(self.window, label='position', from_=-2*np.pi, to=2*np.pi, orient='horizontal', length=400, resolution=0.01)
        self.pos_slider.set(feedback.position[x])
        self.pos_slider.grid(row=1, column=1)
        
        self.vel_slider = tk.Scale(self.window, label='velocity', from_=-2*np.pi, to=2*np.pi, orient='horizontal', length=400, resolution=0.01)
        self.vel_slider.set(feedback.velocity[x])
        self.vel_slider.grid(row=2, column=1)

        self.eff_slider = tk.Scale(self.window, label='effort', from_=-2*np.pi, to=2*np.pi, orient='horizontal', length=400, resolution=0.01)
        self.eff_slider.set(feedback.effort[x])
        self.eff_slider.grid(row=3, column=1)

    def add_dropdown(self):
        self.tkvar = tk.StringVar(self.window)
        self.tkvar.set(self.robot.module_names[0])
        tk.OptionMenu(self.window, self.tkvar, *dict.fromkeys(self.robot.module_names)).grid(row=0, column=1)
        

    def connect_button(self):
        # if self.status == 'Not Connect':
        tk.Button(self.window, text='Connect', command=self.robot.connect).grid(row=1, column=0)

    def test_bt(self):
        self.status = 'Connecting'
        print('ok')

    def stt(self):
        if self.robot.isConnected():
            self.status = 'Connecting'
        else:
            self.status = 'Not Connect'
        tk.Label(self.window, text=self.status).grid(row=0, column=0)


    def begin_canvas(self, h, w):
        self.canvas = tk.Canvas(self.window, height=h, width=w)
        self.canvas.grid(row=1, column=1)

    def clear_canvas(self):
        self.canvas.delete(tk.ALL)

    def update(self):
        # self.clear_canvas()
        self.stt()
        x = self.robot.module_names.index(self.tkvar.get())
        # print(x)
        group_command = hebi.GroupCommand(self.robot.group.size)
        p = []
        for i in range(self.robot.group.size):
            p.append(0)
        
        p[x] = self.pos_slider.get()
        v = []
        for i in range(self.robot.group.size):
            v.append(0)
        v[x] = self.robot.fq*self.vel_slider.get()

        e = []
        for i in range(self.robot.group.size):
            e.append(0)
        e[x] = self.robot.inertia*self.robot.fq*self.robot.fq*self.eff_slider.get()
        
        group_command.position = p
        group_command.velocity = v
        group_command.effort = e
        self.robot.group.send_command(group_command)
        
    def clock(self):
        self.update()
        self.window.after(16, self.clock)



if __name__ == "__main__":
    gui = App("test")
    # gui.begin_canvas(500, 500)
    # gui.stt()
    gui.connect_button()
    gui.add_dropdown()
    gui.add_sliders()
    # gui.bind_keys()
    gui.clock()
    gui.window.mainloop()
