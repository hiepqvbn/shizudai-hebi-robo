import tkinter as tk
import hebi
from time import sleep
import numpy as np
from hebi_arm import RobotArm
import threading


class Robot:
    fq_hz = 0.5  # Hz
    fq = fq_hz*2*np.pi
    inertia = 0.1  # kg.m^2

    def __init__(self):
        self.connect()

    def connect(self):
        self.lookup = hebi.Lookup()
        sleep(2)
        print('connecting')
        if self.isConnected():
            self.family_names = ['Arm']  # self.familyName()
            self.module_names = ['J3_elbow', 'J1_base',
                                 'J2_shoulder', 'gripperSpool']  # self.moduleName()
            print(self.family_names)
            print(self.module_names)
            self.group = self.lookup.get_group_from_names(
                self.family_names, self.module_names)
            self.group.feedback_frequency = 24
        else:
            self.module_names = ['a', 'b', 'c']

    def isConnected(self):
        entries = []
        for entry in self.lookup.entrylist:
            entries.append(entry)
        if entries:
            return True
        else:
            return False

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
        self.robot = RobotArm()
        # print("run time") 
        self.init_xyz = self.robot.get_finger_position(self.robot.group_fbk.position)
        print(type(self.init_xyz))
        self.control_stt = False
        self.is_pressed = False
        self.stt()
        

    def add_sliders(self):
        feedback = self.robot.group.get_next_feedback()
        x = self.robot.module_names.index(self.tkvar.get())

        self.pos_slider = tk.Scale(self.window, label='position', from_=-2 *
                                   np.pi, to=2*np.pi, orient='horizontal', length=400, resolution=0.01)
        self.pos_slider.set(feedback.position[x])
        self.pos_slider.grid(row=1, column=1)

        self.vel_slider = tk.Scale(self.window, label='velocity', from_=-2 *
                                   np.pi, to=2*np.pi, orient='horizontal', length=400, resolution=0.01)
        self.vel_slider.set(feedback.velocity[x])
        self.vel_slider.grid(row=2, column=1)

        self.eff_slider = tk.Scale(self.window, label='effort', from_=-2 *
                                   np.pi, to=2*np.pi, orient='horizontal', length=400, resolution=0.01)
        self.eff_slider.set(feedback.effort[x])
        self.eff_slider.grid(row=3, column=1)

    def add_dropdown(self):
        self.tkvar = tk.StringVar(self.window)
        self.tkvar.set(self.robot.module_names[0])
        tk.OptionMenu(self.window, self.tkvar, *
                      dict.fromkeys(self.robot.module_names)).grid(row=0, column=1)

    def connect_button(self):
        # if self.status == 'Not Connect':
        tk.Button(self.window, text='Connect',
                  command=self.robot.connect).grid(row=1, column=0)

    def control_button(self):
        # if self.status == 'Not Connect':
        tk.Button(self.window, text='Control',
                  command=self.control).grid(row=0, column=1)
        

    def control(self):
        # print("OK")
        self.control_stt = True
        # self.robot.update_end_effector()
        
        # print(self.target_xyz)
        # print(self.control_stt)
        tk.Label(self.window, text="Controling...").grid(row=1, column=1)
        self.begin_canvas(400, 600)
        

    def test_bt(self):
        self.status = 'Connecting'
        print('ok')

    def stt(self):
        if self.robot.isConnected():
            self.status = 'Connecting'
            # self.robot.update_end_effector()
            self.current_xyz = self.robot.finger_pos
        else:
            self.status = 'Not Connect'
        tk.Label(self.window, text=self.status).grid(row=0, column=0)

    def begin_canvas(self, h, w):
        self.canvas = tk.Canvas(self.window, height=h, width=w,  bg = '#afeeee')
        self.canvas.grid(row=2, column=1)

    def clear_canvas(self):
        self.canvas.delete(tk.ALL)

    def key_press(self, event):
        self.is_pressed = True
        if self.control_stt and self.status == 'Connecting':
            # print("controling...")
            print("key pressed: "+event.char)
        # if event.char == 'a':
            if event.char=='a': #<--
                self.target_xyz[0] -=0.01
                # self.robot.make_robot_trajectory(self.target_xyz)
            if event.char=='d': #-->
                self.target_xyz[0] +=0.01
                # self.robot.make_robot_trajectory(self.target_xyz)
            if event.char=='w': #y+++
                self.target_xyz[1] +=0.01
                # self.robot.make_robot_trajectory(self.target_xyz)
            if event.char=='s': #y---
                self.target_xyz[1] -=0.01
                # self.robot.make_robot_trajectory(self.target_xyz)
            if event.char=='q': #z---
                self.target_xyz[2] -=0.01
                # self.robot.make_robot_trajectory(self.target_xyz)
            if event.char=='e': #z+++
                self.target_xyz[2] +=0.01
                # self.robot.make_robot_trajectory(self.target_xyz)

    def key_release(self, event):
        if self.control_stt and self.status == 'Connecting':
            print("release "+ event.char)
            self.robot.make_robot_trajectory(self.target_xyz)
            self.is_pressed = False
    # binds keys to move arm

    def bind_keys(self):
        # self.window.bind("<Left>", lambda event, move=(-1, 0, 0), angle=0: self.move_camera(move, angle))
        # self.window.bind("<Right>", lambda event, move=(1, 0, 0), angle=0: self.move_camera(move, angle))
        # self.window.bind("<Up>", lambda event, move=(0, 1, 0), angle=0: self.move_camera(move, angle))
        # self.window.bind("<Down>", lambda event, move=(0, -1, 0), angle=0: self.move_camera(move, angle))
        # self.window.bind("<w>", lambda event, move=(0, 0, 1), angle=0: self.move_camera(move, angle))
        # self.window.bind("<s>", lambda event, move=(0, 0, -1), angle=0: self.move_camera(move, angle))
        # self.window.bind("<e>", lambda event, move=(0, 0, 0), angle=-0.314: self.move_camera(move, angle))
        # self.window.bind("<d>", lambda event, move=(0, 0, 0), angle=0.314: self.move_camera(move, angle))
        # key_tracker = KeyTracker()
        # key_tracker.track('c')
        # self.window.bind('<KeyPress>', key_tracker.report_key_press)
        # self.bind('<KeyRelease>', key_tracker.report_key_release)
        
        self.window.bind("<KeyPress>", self.key_press)
        self.window.bind("<KeyRelease>", self.key_release)
        

    def update(self):
        # print(self.init_xyz)
        
        self.stt()

        # x = self.robot.module_names.index(self.tkvar.get())
        # # print(x)
        # group_command = hebi.GroupCommand(self.robot.group.size)
        # p = []
        # for i in range(self.robot.group.size):
        #     p.append(0)

        # p[x] = self.pos_slider.get()
        # v = []
        # for i in range(self.robot.group.size):
        #     v.append(0)
        # v[x] = self.robot.fq*self.vel_slider.get()

        # e = []
        # for i in range(self.robot.group.size):
        #     e.append(0)
        # e[x] = self.robot.inertia*self.robot.fq * \
        #     self.robot.fq*self.eff_slider.get()

        # group_command.position = p
        # group_command.velocity = v
        # group_command.effort = e
        # self.robot.group.send_command(group_command)
        # if self.control_stt and self.status == 'Connecting':
        #     self.clear_canvas()
        #     self.canvas.create_text(240,20,fill="darkblue",font="Times 20 italic bold", text=self.current_xyz)
        #     self.canvas.create_text(240,50,fill="darkblue",font="Times 20 italic bold", text=self.target_xyz)
        #     if not self.is_pressed:
        #         # pass
        #         print("loop")
        #         self.robot.keep_position()

    def clock(self):
        self.update()
        self.window.after(16, self.clock)


class KeyTracker():
    key = ''
    last_press_time = 0
    last_release_time = 0

    def track(self, key):
        self.key = key

    def is_pressed(self):
        return time.time() - self.last_press_time < .1

    def report_key_press(self, event):
        if event.keysym == self.key:
            if not self.is_pressed():
                on_key_press(event)
            self.last_press_time = time.time()

    def report_key_release(self, event):
        if event.keysym == self.key:
            timer = threading.Timer(.1, self.report_key_release_callback, args=[
                                    event])
            timer.start()

    def report_key_release_callback(self, event):
        if not self.is_pressed():
            on_key_release(event)
        self.last_release_time = time.time()


if __name__ == "__main__":
    gui = App("test")
    # gui.begin_canvas(500, 500)
    # gui.stt()
    gui.connect_button()
    gui.control_button()
    # gui.add_dropdown()
    # gui.add_sliders()
    
    gui.bind_keys()
    
    gui.clock()
    gui.window.mainloop()
