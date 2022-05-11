import time
import numpy as np
import hebi
import signal
from pygame_GUI import pygameGUI
from xbox360controller import Xbox360Controller
import multiprocessing as mp
from hebi_arm import RobotArm


class Controller(Xbox360Controller):
    def __init__(self, index=0, axis_threshold=0.2, raw_mode=False):
        try:
            super().__init__(index, axis_threshold, raw_mode)
            self._isConnected = True
        except:
            self._isConnected = False
        self.processes = []
        print('_----')
        self.gui = pygameGUI().start()
        self.robot = RobotArm()
        # print(self.robot.isConnected)

    @property
    def isConnected(self):
        return self._isConnected

    def test(self):
        time.sleep(5)
        print("AAA")

    def on_button_pressed(self, button):
        # self.gui.step()
        print('Button {0} was pressed'.format(button.name))
        
        if button.name is 'button_a':
            p = mp.Process(target=self.test)
            self.processes.append(p)
            p.start()
            print('Done')
            
        if button.name is 'button_b':
            for p in self.processes:
                if p.is_alive():
                    p.kill()
            print('ok')

    def on_button_released(self, button):
        print('Button {0} was released'.format(button.name))


    def on_axis_moved(self, axis):
        print('Axis {0} moved to {1} {2}'.format(axis.name, axis.x, axis.y))


def main():
    controller = Controller()
    # print('here')
    # print(controller.get_available())

    if controller.isConnected:
        try:
            with controller:
                print(controller.name)

                #Button A events
                controller.button_a.when_pressed = controller.on_button_pressed

                controller.button_b.when_pressed = controller.on_button_pressed

                signal.pause()


        except KeyboardInterrupt:
            pass

# def main(self):
    
#     controller.main()

if __name__=="__main__":
    main()
    time.sleep(3)
    # controller = Controller()
    # controller.main()
    # p = Process(target=controller.main, daemon=True)
    # p.start()
    # with controller:
    #     if controller.button_b.is_pressed:
    #         print('ok')
    #         # if p.is_alive():
    #         #     p.kill()
    #     signal.pause()