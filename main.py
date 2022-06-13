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

if __name__=='__main__':
    GUI()