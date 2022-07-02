from enum import Enum
import math
from queue import Queue
from collections import namedtuple
import pygame
import time
import multiprocessing as mp
import threading
from Arm.hebi_arm import RobotArm
from Arm.xbox_controller import Controller

SPEED = 40

RobotEvent = namedtuple(
    'Robot',['is_pause']
)

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

pygame.init()

class pygameGUI():
    def __init__(self, w=640, h=480, camera=None) -> None:
        self.w = w
        self.h = h
        # super(pygameGUI, self).__init__()

        self.robot_com_queue = Queue()
        self.robot_event = threading.Event()
        self.robot = RobotArm(self.robot_com_queue, self.robot_event)
        self.robot.start()
        # print("check here")
        self.controller = Controller()
        if self.controller.hasBeenConnected:
            self._set_controller_func()
            # time.sleep(2)
            self.controller.set_rumble(0.8, 0.8, 1500)
            
        # init display
    # def run(self, a = None):
        # print(a)
        print("Starting GUI...")
        # time.sleep(18)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Controller')
        self.font = pygame.font.SysFont('arial', 25)
        self.clock = pygame.time.Clock()
        self.reset()
        
        print("GUI done")

    # class RobotMode(Enum):
    #     POSITION = 0
    #     VELOCITY = 1
    #     EFFORT = 2

    def reset(self):
        self.controller_mode = False    #Actuator ;     True is Endeffector 
        self.robot_mode = self.robot.RobotMode.POSITION
        self.joint_l_mode = self.robot.Actuator.J1_base
        self.joint_r_mode = self.robot.Actuator.J2_shoulder
        self.iter = 0
        self.pressed_button = None
        self.Estop_stt = False

    def _set_controller_func(self):
        for button in self.controller.buttons:
            button.when_pressed = self.on_button_pressed
            button.when_released = self.on_button_released
        for axis in [self.controller.axis_l, self.controller.axis_r]:
            axis.when_moved = self.on_axis_moved
        
    #loop here
    def step(self, button=None):
        # print('GUI step')
        # self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # if self.controller.get_event():
        #     print("this ",self.controller.get_event())
        if not self.controller.isConnected:
            self.controller.recheck()
            if self.controller.isConnected:
                self._set_controller_func()
                self.controller.set_rumble(0.8, 0.8, 1500)

        # if not self.robot.isConnected:
        #     if not self.robot.is_alive():
        #         self.robot.start()

        # print('robot thread is {}'.format(self.robot.is_alive()))
        
        # if button:
        #     print("ok")
        if self.pressed_button == 'E-Stop':
            self._update_ui()
            self.clock.tick(SPEED)
            while True:
                if self.pressed_button == 'button_start':
                    self.reset()
                    self.send_robot('reset')
                    break
        
        # update ui and clock
        self.iter += 1
        self._update_ui()
        self.clock.tick(SPEED)
       

    def _update_ui(self):
        
        self.display.fill(BLACK)
        # time.sleep(5)
        # print("in")
        # if self.iter%2 == 0:
        #     pygame.draw.rect(self.display, WHITE, pygame.Rect(self.w/3, self.h/3, self.w/3, self.h/3))
        # else:
        #     pygame.draw.rect(self.display, BLACK, pygame.Rect(self.w/3, self.h/3, self.w/3, self.h/3))
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.w/20, self.h/20, self.w*9/10, self.h*9/10))

        text = self.font.render(self.pressed_button, True, BLUE1)
        self.display.blit(text, [self.w*7/20,self.h*13/20])
        # controller_text = "Connected" if self.controller.isConnected else "Disconnect"
        controller_text = self.font.render("Controller: " + ("Connected" if self.controller.isConnected else "Disconnect"), True, BLUE1)
        self.display.blit(controller_text, [self.w/20,self.h/20])
        #HEBI Arm connection status
        robot_text = self.font.render("HEBI: " + ("Connected" if self.robot.isConnected else "Disconnect"), True, RED)
        self.display.blit(robot_text, [self.w/20,self.h*2/20])
        #MODE display
        if self.controller.isConnected:
            mode_text = self.font.render("MODE: " + ("End effector" if self.controller_mode else "Actuators"), True, BLUE2)
            self.display.blit(mode_text, [self.w*12/20,self.h/20])
            if not self.controller_mode:
                robot_mode_text = self.font.render(self.robot_mode.name, True, BLUE2)
                self.display.blit(robot_mode_text, [self.w*15/20,self.h*2/20])
                joint_l_mode_text = self.font.render(self.joint_l_mode.name, True, BLUE2)
                self.display.blit(joint_l_mode_text, [self.w*5/20,self.h*9/20])
                joint_r_mode_text = self.font.render(self.joint_r_mode.name, True, BLUE2)
                self.display.blit(joint_r_mode_text, [self.w*13/20,self.h*9/20])
        pygame.display.flip()

    def send_robot(self, mes):
        # if self.robot_com_queue.empty():
        self.robot_com_queue.put(mes)
        time.sleep(0.01)
        self.robot_event.set()
        
    def on_button_pressed(self, button):
        
        # print('Button {0} was pressed'.format(button.name))
        self.pressed_button = button.name

        if button.name == 'button_a' and self.controller.button_trigger_l.is_pressed:
            self.pressed_button = 'E-Stop'
            self.controller.set_rumble(1.0, 1.0, 100)
            exit(1)
            self.send_robot('pause')
            
            
        if button.name == 'button_x':
            self.send_robot = 'set'

        # if button.name == 'button_b':
        #     self.controller.set_rumble(0.7, 0.7, 300)
        if button.name == 'button_trigger_r':
            self.controller_mode = False if self.controller_mode else True
            self.robot_mode = self.robot.RobotMode.POSITION if self.controller_mode else self.robot_mode  

        # Actuators mode
        if not self.controller_mode:
            #switch between position, velocity, effort(easy-version: only control position)
            if button.name == 'button_y':
                v=(self.robot_mode.value + 1)%len(self.robot.RobotMode)
                for r in self.robot.RobotMode:                        
                    self.robot_mode = r if r.value == v else self.robot_mode
            #choose the actuator for left thumb
            ###option 1:    fix left joystick for only J1_base
            if button.name == 'button_thumb_l':
                #########################################################
                # v=(self.joint_l_mode.value +1)%len(self.robot.Actuator)
                # if v == self.joint_r_mode.value:
                #     v = (v +1)%len(self.robot.Actuator)
                # for j in self.robot.Actuator:
                #     self.joint_l_mode = j if j.value == v else self.joint_l_mode
                #########################################################
                pass
            #choose the actuator for right thumb
            ###option 1:    set right joystick for only 2 mode
            if button.name == 'button_thumb_r':
                #v=(self.joint_r_mode.value +1)%len(self.robot.Actuator)
                # if v == self.joint_l_mode.value:
                #     v = (v +1)%len(self.robot.Actuator)
                
                v = 2 if self.joint_r_mode.value == 1 else 1 # v in {1,2}
                for j in self.robot.Actuator:
                    self.joint_r_mode = j if j.value == v else self.joint_r_mode
                
        #End-effector mode(Inverse kimetic)    
        if self.controller_mode:
            pass        

    def on_button_released(self, button):
        # print('Button {0} was released'.format(button.name))
        self.pressed_button = None


    def on_axis_moved(self, axis):
        #run on the same thread with controller
        # print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))
        
        if not (self.controller.button_thumb_l.is_pressed or self.controller.button_thumb_r.is_pressed): 
            if self.robot.isConnected:
                if not self.controller_mode:
                    # angle = math.atan2(axis.x, axis.y)
                    # print("angle of jointstick is {}".format(angle))
                    # if self.controller.button_x.is_pressed:
                    if axis.name == 'axis_l':
                        self.send_robot((self.controller_mode, self.robot_mode, self.joint_l_mode, axis.x))
                    if axis.name == 'axis_r':
                        self.send_robot((self.controller_mode, self.robot_mode, self.joint_r_mode, axis.y))


def main():
    gui = pygameGUI()
    while True:
        gui.step()

if __name__=="__main__":
    # gui = pygameGUI()
    # t1 = mp.(target=check)
    # t1.start()
    main()
    # pass
