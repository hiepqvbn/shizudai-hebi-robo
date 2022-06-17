from enum import Enum
from logging import raiseExceptions
import threading
import hebi
import numpy as np
from time import sleep, time
import keyboard
import os
# import grip

class RobotArm(threading.Thread):

  AMP = 1.0
  SPEED = 0.01   #set actuator's speed as constant 0.1 rad/s
  EPS = 0.01 
  LIMIT_POSITION = {
    "J1_base": {"max":None, "min":None},
    "J2_shoulder": {"max":None, "min":None},
    "J3_elbow": {"max":None, "min":None},
  }

  def __init__(self, queue=None, event=None):
    # sleep(3)
    self.gui_queue = queue
    self.gui_event = event
    self._isConnected = False
    super(RobotArm, self).__init__()

  def run(self):
    #HEBI Robot Thread

    #Connect to HEBI
    self.connect()

    #HEBI Connected situation
    if self.isConnected:  #connected to robot --> run below command in other thread
      print("HEBI connected!!")
      self.target_position = self.joint_angles
      self.vel = 0
      self.mode = None
      

      while True: #Run a loop --- Main program for HEBI is HERE

        self.refresh_fbk()  #Need to refresh feedback every loop to get update feedback(position, velocity, effort)
        gui_mes = self.get_event(self.gui_event)  #Get message from Queue

        # Send command every loop
        self.update_joint_angles(self.vel, self.mode)
        self.joint_angles[self.mode] += self.vel*self.SPEED
        self.set_arm_position(self.joint_angles)

        # Check Controller Message            
        if gui_mes: #Some buttons is pressed
          # print(gui_mes)
          """
          # if gui_mes == 'set':
          #   self.refresh_fbk()
          #   self.joint_angles = self.group_fbk.position

          # if gui_mes == 'pause':
          #   self.cancel_all_command()
          #   while not gui_mes == 'reset':
          #     gui_mes = self.get_event(self.gui_event)
          #   print(gui_mes)
          #   continue
          """

          #robot controller mode is each actuator gui_mes[0]==False
          if not gui_mes[0]:
            #set position for robot arm
            if gui_mes[1] == self.RobotMode.POSITION:             
              self.vel = gui_mes[3]
              self.mode=gui_mes[2]
              print(self.vel)
              if np.abs(self.vel)<0.8:
                self.vel = 0

    
    #else: couldn't connect to robot --> thread done
    

  def connect(self):
    '''
    Look up HEBI then initialize group, feedback etc.
    '''
    print('Looking for HEBI...')
    self.lookup = hebi.Lookup()
    sleep(0.1) #wait for look-up hebi(stable)

    # Modules selection 
    self.families = ['Arm', 'Arm', 'Arm', 'Arm', 'Arm']  #, 'Arm''X5-1'
    self.names = ['J1_base', 'J2_shoulder', 'J3_elbow', 'J4_wrist1', 'J5_wrist2']  #, 'gripperSpool', X-01069
    self.group = self.lookup.get_group_from_names(self.families, self.names)

    if self.group:
      self._isConnected = True
      self.group.feedback_frequency = 24 #feedback refresh frequency is 24Hz
      self.arm = self.robot_model()
      self.num_joints = self.group.size
      # print(self.num_joints)
      self.group_fbk = self.robot_fbk()
      self.joint_angles = self.group_fbk.position #get positions of each actuator at the first time to avoid suddenly move(dangerous)
      print("position is {}".format(type(self.joint_angles)))
      self.finger_pos = self.get_finger_position(self.joint_angles)
      self.group_command = hebi.GroupCommand(self.num_joints)
    else:
      self._isConnected = False
      print('Could not connect to HEBI robot!!!')
  
  class Actuator(Enum):
    J1_base = 0
    J2_shoulder = 1
    J3_elbow = 2
    J4_wrist1 = 3
    J5_wrist2 = 4

  class RobotMode(Enum):
    POSITION = 0
    VELOCITY = 1
    EFFORT = 2

  #only check for the first time connect so make sure the robot connect all the time.
  #if after that robot is disconnect then we need to reset the program
  @property   
  def isConnected(self):
    return self._isConnected
    
    
  def robot_model(self):
    """
    make HEBI model from hrdf file for kinematics/inverse-kinematics function
    """
    hrdf_filename = "A-2085-05.hrdf"
    try:
      dir = os.path.abspath( os.path.dirname( __file__ ) )
      return hebi.robot_model.import_from_hrdf(os.path.join(dir,hrdf_filename))
    except:
      print("Could not load HRDF.")
      exit(1)

  def robot_fbk(self):
    group_fbk = hebi.GroupFeedback(self.num_joints)
    if self.group.get_next_feedback(reuse_fbk=group_fbk) is None:
      print("Couldn't get feedback.")
      self._isConnected = False
      exit(1)
    return group_fbk

  def refresh_fbk(self,duration=5):
    """
    Control the rate of func = the group feedback frequency\n
    If cannot get the feedback in duration(default=5s) --> robot is disconnected
    """
    
    start_time = time()
    while (time()-start_time)<duration:
      fbk = self.group_fbk
      self.group_fbk = self.group.get_next_feedback(reuse_fbk=self.group_fbk)
      if self.group_fbk is None:
        self.group_fbk = fbk
        continue
      else:
        break
    print("Refresh time is {}".format(time()-start_time))
    if self.group_fbk is None:
      self._isConnected = False
      print("HEBI disconnected!!!")
    
    
  """
  # def feedback_handler(self):
  #   self.angles = self.group_fbk.position
  #   self.transform = self.arm.get_end_effector(self.angles)
  #   print('x,y,z: {0}, {1}, {2}'.format(self.transform[0, 3], self.transform[1, 3], self.transform[2, 3]))
  """

  def get_event(self, e):
    # event_is_set = e.wait(0.01)
    # e.clear()
    sleep(0.01)
    if not self.gui_queue.empty():
      return self.gui_queue.get()
    else:
      return

  def actuator_command(self, joint, mode, signal):
    # if signal <=1 or signal>= -1:
    
    self.refresh_fbk()
    if mode == self.RobotMode.POSITION:
      # positions = np.zeros((self.num_joints, 2), dtype=np.float64)
      current_pos = self.group_fbk.position
      current_pos[joint.value] += self.AMP*signal
      self.group_command.position = current_pos
      self.group.send_command(self.group_command)

    # else:
    #   print("Input signal is unvalid")

  def get_finger_position(self, joint_angles):
    
    return self.arm.get_end_effector(joint_angles)[0:3,3]

  def update_end_effector(self):
    self.refresh_fbk()
    self.joint_angles = self.group_fbk.position
    self.finger_pos = self.get_finger_position(self.joint_angles)
    
  def IK_solute(self, target_pos):
    self.refresh_fbk()
    init_joint_angles = self.group_fbk.position
    # Note: this is a numerical optimization and can be significantly affected by initial conditions (seed joint angles)
    ee_pos_objective = hebi.robot_model.endeffector_position_objective(target_pos)
    ik_result_joint_angles = self.arm.solve_inverse_kinematics(init_joint_angles, ee_pos_objective)
    return ik_result_joint_angles

  def make_robot_trajectory(self, target_pos):
    positions = np.zeros((self.num_joints, 2), dtype=np.float64)
    self.refresh_fbk()
    current_pos = self.group_fbk.position
    positions[:, 0] = current_pos
    positions[:, 1] = self.IK_solute(target_pos)

    self.time_vector = [0, 1]
    self.trajectory = hebi.trajectory.create_trajectory(self.time_vector, positions)
    duration = self.trajectory.duration
    start = time()

    t = time() - start

    while t < duration:
      self.refresh_fbk()
      t = time() - start
      pos, vel, acc = self.trajectory.get_state(t)
      self.group_command.position = pos
      self.group_command.velocity = vel
      self.group.send_command(self.group_command)

  def cancel_all_command(self):
    self.group_command.position[:] = np.nan
    self.group_command.velocity[:] = np.nan
    self.group_command.effort[:] = np.nan
    self.group.send_command(self.group_command)

  def update_joint_angles(self, vel, mode):
    self.joint_angles[mode.value] += vel*self.SPEED
    if self.LIMIT_POSITION[mode]["min"]: #not None
      if self.joint_angles[mode.value]<self.LIMIT_POSITION[mode]["min"]:
        self.joint_angles[mode.value]=self.LIMIT_POSITION[mode]["min"]
    if self.LIMIT_POSITION[mode]["max"]: #not None
      if self.joint_angles[mode.value]>self.LIMIT_POSITION[mode]["max"]:
        self.joint_angles[mode.value]=self.LIMIT_POSITION[mode]["max"]

  def set_arm_position(self, position):
    """
    Set HEBI Arm position quickly
    """
    self.group_command.position = position
   
    self.group.send_command(self.group_command)

  def reach_position(self, target, speed=SPEED):
    # while True:
    self.refresh_fbk()
    self.joint_angles = self.group_fbk.position   #update self.joint_angles
    next_position = np.where(np.absolute(target - self.joint_angles)<self.EPS,0,speed*(target - self.joint_angles)/np.absolute(target - self.joint_angles))
    self.group_command.velocity = next_position
    
    self.group.send_command(self.group_command)





# group.add_feedback_handler(feedback_handler)
# group.feedback_frequency = 10.0 # Prevent printing to the screen too much

# # Control the robot at 10 Hz for 30 seconds
# sleep(30)

# target_xyzo = [0.2, 0.1, 0.6]
# def move_finger(target_xyz):

# Note: user should check if the positions are appropriate for initial conditions.
# initial_joint_angles = group_fbk.position
# print(initial_joint_angles)

################################################################
# Get IK Solution with one objective
################################################################

# Just one objective:

# print('Target position: {0}'.format(target_xyzo))
# print('IK joint angles: {0}'.format(ik_result_joint_angles))
# print('FK of IK joint angles: {0}'.format(arm.get_end_effector(ik_result_joint_angles)[0:3, 3]))

################################################################
# Send commands to the physical robot
################################################################

# Move the arm
# Note: you could use the Hebi Trajectory API to do this smoothly
# group_command = hebi.GroupCommand(group.size)
# group_command.position = ik_result_joint_angles


# positions = np.zeros((num_joints, 2), dtype=np.float64)
# current_pos = group_feedback.position
# print(current_pos)

# positions[:, 0] = current_pos
# positions[:, 1] = ik_result_joint_angles

# time_vector = [0, 3]
# trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
# group_command = hebi.GroupCommand(num_joints)
# duration = trajectory.duration

# start = time()
# t = time() - start

# while t < duration:
#   # Serves to rate limit the loop without calling sleep
#   group.get_next_feedback(reuse_fbk=group_feedback)
#   t = time() - start

#   pos, vel, acc = trajectory.get_state(t)
#   group_command.position = pos
#   group_command.velocity = vel
#   group.send_command(group_command)

# while True:
#   group_command.position = ik_result_joint_angles
#   group.send_command(group_command)

# arm = RobotArm()
# target_xyz = arm.finger_pos
if __name__=="__main__":
  arm = RobotArm()
  arm.run()
  print(arm.isConnected)
  # initialize()
#   last_time = time()
#   while True:
#     # group.get_next_feedback(reuse_fbk=group_feedback)
#       # # Collect events until released
    
#     if keyboard.is_pressed('f'):
#       grip.stt=1
#       grip.gripped(grip.stt)
#     if keyboard.is_pressed('j'):
#       grip.stt=0
#       grip.gripped(grip.stt)
    
#     loop_time = time() - last_time
#     last_time=time()
#     # print(loop_time)
#     print(target_xyz)
#     # sleep(0.005)
#     if keyboard.is_pressed('a'):
#       target_xyz[0] -=0.01
#       arm.make_robot_trajectory(target_xyz)
#     if keyboard.is_pressed('d'):
#       target_xyz[0] +=0.01
#       arm.make_robot_trajectory(target_xyz)
#     if keyboard.is_pressed('w'):
#       target_xyz[1] +=0.01
#       arm.make_robot_trajectory(target_xyz)
#     if keyboard.is_pressed('s'):
#       target_xyz[1] -=0.01
#       arm.make_robot_trajectory(target_xyz)
#     if keyboard.is_pressed('q'):
#       target_xyz[2] -=0.01
#       arm.make_robot_trajectory(target_xyz)
#     if keyboard.is_pressed('e'):
#       target_xyz[2] +=0.01
#       arm.make_robot_trajectory(target_xyz)
#     if keyboard.is_pressed('esc'):
#       break
# # for i in range(100):
# #   # group.send_command(group_command)
# #   # Note: the arm will go limp after the 100 ms command lifetime,
# #   # so the command is repeated every 50 ms for 5 seconds.
# #   sleep(0.05)
