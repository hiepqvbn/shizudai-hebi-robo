import hebi
import numpy as np
from time import sleep, time
import keyboard
# import grip

class RobotArm():
  def __init__(self):
    # sleep(3)
    print('Robot init')
    self._isConnected = False
    self.connect()
    
    

  def connect(self):
    self.lookup = hebi.Lookup()
    sleep(2)
    # print(self.lookup.entrylist)
    # if self.lookup.entrylist:
      
    self.families = ['Arm', 'Arm', 'Arm', 'Arm']  #, 'Arm''X5-1'
    self.names = ['J1_base', 'J2_shoulder', 'J3_elbow', 'J4_wrist']  #, 'gripperSpool', X-01069
    self.group = self.lookup.get_group_from_names(self.families, self.names)
    if self.group:
      self._isConnected = True
      self.group.feedback_frequency = 24
      self.arm = self.robot_model()
      self.num_joints = self.group.size
      self.group_fbk = self.robot_fbk()
      self.joint_angles = self.group_fbk.position
      self.finger_pos = self.get_finger_position(self.joint_angles)
      self.group_command = hebi.GroupCommand(self.num_joints)
    else:
      self._isConnected = False
      print('Could not connect to HEBI robot!!!')
  
  @property
  def isConnected(self):
    return self._isConnected
    # group_fbk = hebi.GroupFeedback(self.num_joints)
    # if self.group.get_next_feedback(reuse_fbk=group_fbk) is None:
    #   return False
    # else: return True
    # entries = []
    # for entry in self.lookup.entrylist:
    #     entries.append(entry)
    # if entries: return True
    # else: return False

  def robot_model(self):
    try:
      return hebi.robot_model.import_from_hrdf("A-2085-04G.hrdf")
    except:
      print("Could not load HRDF.")
      exit(1)

  def robot_fbk(self):
    group_fbk = hebi.GroupFeedback(self.num_joints)
    if self.group.get_next_feedback(reuse_fbk=group_fbk) is None:
      print("Couldn't get feedback.")
      exit(1)
    return group_fbk

  def refresh_fbk(self):
    self.group.get_next_feedback(reuse_fbk=self.group_fbk)
    

  # def feedback_handler(self):
  #   self.angles = self.group_fbk.position
  #   self.transform = self.arm.get_end_effector(self.angles)
  #   print('x,y,z: {0}, {1}, {2}'.format(self.transform[0, 3], self.transform[1, 3], self.transform[2, 3]))

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

  def keep_position(self):
    # while True:
    self.refresh_fbk()
    current_pos = self.group_fbk.position
    self.group_command.position = current_pos
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
