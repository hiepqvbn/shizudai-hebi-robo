import pygame
import time
import multiprocessing as mp
import threading
from hebi_arm import RobotArm
from xbox_controller import Controller

SPEED = 40

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

pygame.init()

class pygameGUI():
    def __init__(self, w=640, h=480) -> None:
        self.w = w
        self.h = h
        # super(pygameGUI, self).__init__()
        self.robot = RobotArm()
        self.robot.start()
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

    def reset(self):
        self.iter = 0
        self.pressed_button = None
        self.Estop_stt = False

    def _set_controller_func(self):
        for button in self.controller.buttons:
            button.when_pressed = self.on_button_pressed
            button.when_released = self.on_button_released
        for axis in [self.controller.axis_l, self.controller.axis_r]:
            axis.when_moved = self.on_axis_moved
        

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
        if self.controller.isConnected:
            # print('controller is connected')
            # with self.controller as c:
                # print('in with')
            pass
                # if 1:
                #     print('this ok ')
                # if c.button_a.is_pressed:
                #         print('Button A')
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_a:
            #         pygame.quit()
            #         quit()
        else:
            # print('controller is not connected')
            self.controller.recheck()
            if self.controller.isConnected:
                self._set_controller_func()
                # time.sleep(10)
                self.controller.set_rumble(0.8, 0.8, 1500)
        
        if button:
            print("ok")
        # 2. move
        # self._move(action) # update the head
        # self.snake.insert(0, self.head)
        
        # 3. check if game over
        # reward = 0
        # game_over = False
        # if self.is_collision() or self.frame_iteration > 100*len(self.snake):
        #     game_over = True
        #     reward = -10
        #     return reward, game_over, self.score

        # 4. place new food or just move
        # if self.head == self.food:
        #     self.score += 1
        #     reward = 10
        #     self._place_food()
        # else:
        #     self.snake.pop()
        
        # 5. update ui and clock
        self.iter += 1
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score

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
        self.display.blit(text, [self.w/2,self.h/2])
        # controller_text = "Connected" if self.controller.isConnected else "Disconnect"
        controller_text = self.font.render("Controller: " + ("Connected" if self.controller.isConnected else "Disconnect"), True, BLUE1)
        self.display.blit(controller_text, [self.w/20,self.h/20])
        pygame.display.flip()

    def on_button_pressed(self, button):
        
        print('Button {0} was pressed'.format(button.name))
        self.pressed_button = button.name

        if button.name == 'button_a' and self.controller.button_trigger_l.is_pressed:
            self.pressed_button = 'E-Stop'
            self.controller.set_rumble(1.0, 1.0, 700)
        if button.name == 'button_b':
            self.controller.set_rumble(0.7, 0.7, 300)
        

    def on_button_released(self, button):
        print('Button {0} was released'.format(button.name))
        self.pressed_button = None


    def on_axis_moved(self, axis):
        print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))


def check():
    gui = pygameGUI()
    while True:
        gui.step()

if __name__=="__main__":
    # gui = pygameGUI()
    # t1 = mp.(target=check)
    # t1.start()
    check()
    # pass
