import sys



# from computer_vision import detect_blue_mark

argvs = sys.argv

def GUI():
    from Arm import pygame_GUI
    pyGUI = pygame_GUI.pygameGUI()
    while True:
        pyGUI.step()

def collect_data():
    from data_collect import data_collect
    collect = data_collect.DataCollect()
    if len(argvs)==3:
        if argvs[2] == 'clear':
            collect.clear_csv()
    if len(argvs)==2:
        while True:
            try:
                (a, b) = input("Please input 2 value or enter to exit: ").split()
                collect.write_data([int(a), int(b)])
            except:
                break
        collect.save_dataframe()
        print(collect.log_df.head())
        print("Done")
        
if len(argvs)>1:
    if argvs[1] == 'gui':
        GUI()

    if argvs[1] == 'collect_data':
        collect_data()

def main():
    import threading
    from Arm import pygame_GUI
    from data_collect import data_collect
    from computer_vision import realsense_depth, detect_blue_mark
    from data_collect import model
    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt

    class mainGUI(pygame_GUI.pygameGUI):
        def __init__(self, w=640, h=480) -> None:
            super().__init__(w, h)

        #rewrite func for button command
        def on_button_pressed(self, button):
            super().on_button_pressed(button)
            if button.name == 'button_b':
                self.controller.set_rumble(0.7, 0.7, 300)
                # print(self.robot.group_fbk.position)
                collect.write_data_to_csv(self.robot.group_fbk.position)

    # detect_blue_mark.main()
    pyGUI = mainGUI()
    collect = data_collect.DataCollect(cols=pyGUI.robot.names)
    
    camera_thread = threading.Thread(target=detect_blue_mark.main, args=())
    camera_thread.start()
    # rc = realsense_depth.DepthCamera()
    while True:
        pyGUI.step()
    

if len(argvs)==1 and __name__=='__main__':
    main()