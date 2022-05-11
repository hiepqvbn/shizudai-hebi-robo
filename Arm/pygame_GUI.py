import pygame
import time
import multiprocessing as mp
import threading

SPEED = 40

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

pygame.init()

class pygameGUI(threading.Thread):
    def __init__(self, w=640, h=480) -> None:
        self.w = w
        self.h = h
        super().__init__()
        
        
        # init display
    def run(self):
        print("GUI init")
        time.sleep(8)
        print("GUI done")
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Controller')
        self.font = pygame.font.SysFont('arial', 25)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.iter = 0

    def step(self, button=None):
        # self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    pygame.quit()
                    quit()
        
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
        # print("in")
        self.display.fill(BLACK)
        time.sleep(1)
        if self.iter%2 == 0:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(self.w/3, self.h/3, self.w/3, self.h/3))
        else:
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.w/3, self.h/3, self.w/3, self.h/3))
        text = self.font.render("abc", True, BLUE1)
        self.display.blit(text, [self.w/2,self.h/2])
        pygame.display.flip()

def check():
    gui = pygameGUI()
    while True:
        gui.step()

if __name__=="__main__":
    # gui = pygameGUI()
    # t1 = mp.(target=check)
    # t1.start()
    # check()
    pass
