import hebi
from time import sleep, time
from matplotlib import pyplot as plt
from math import cos, pi, sin
import keyboard

lookup = hebi.Lookup()

# Wait 2 seconds for the module list to populate
sleep(2.0)

family_name = "Arm"
module_name = "gripperSpool"

group = lookup.get_group_from_names([family_name], [module_name])

if group is None:
  print('Group not found: Did you forget to set the module family and name above?')
  exit(1)

print('Created group from module {0} | {1}.'.format(family_name, module_name))
group_command  = hebi.GroupCommand(group.size)
group_feedback = hebi.GroupFeedback(group.size)

# # Start logging in the background
# group.start_log('logs', mkdirs=True)

# freq_hz = 0.5                 # [Hz]
# freq    = freq_hz * 2.0 * pi  # [rad / sec]
# amp     = pi * 0.025           # [rad] (45 degrees)

# # Inertia parameters for converting acceleration to torque. This inertia value corresponds
# # to roughly a 300mm X5 link extending off the output.
# inertia = 0.01                # [kg * m^2]

# duration = 4              # [sec]
# start = time()
# t = time() - start

# while True:
  # Even though we don't use the feedback, getting feedback conveniently
  # limits the loop rate to the feedback frequency
#   group.get_next_feedback(reuse_fbk=group_feedback)
#   t = time() - start

#   # Position command
#   group_command.position = amp * sin(freq * t)
#   # Velocity command (time derivative of position)
#   group_command.velocity = freq * amp * cos(freq * t)
#   # Acceleration command (time derivative of velocity)
#   accel = -freq * freq * amp * sin(freq * t)
  # Convert to torque
max_effort=-5

def gripped(stt):
    if stt == 0:
        # current_effort = group_feedback.effort
        # print(current_effort)
        # group_command.effort = current_effort - 0.5
        # if group_command.effort >= max_effort:
        group_command.effort = max_effort
        group.send_command(group_command)
    if stt == 1:
        group_command.effort = 1
        group.send_command(group_command)
# group_command.effort = -2
# group.send_command(group_command)
stt=1
if __name__=="__main__":
  while True:
      # print("run")
      group.get_next_feedback(reuse_fbk=group_feedback)
      # # Collect events until released
      gripped(stt)
      if keyboard.is_pressed('f'):
          stt=1
      if keyboard.is_pressed('j'):
          stt=0
      if keyboard.is_pressed('esc'):
          break
    
# Stop logging. `log_file` contains the contents of the file
# log_file = group.stop_log()

# hebi.util.plot_logs(log_file, 'position', figure_spec=101)
# hebi.util.plot_logs(log_file, 'velocity', figure_spec=102)
# hebi.util.plot_logs(log_file, 'effort', figure_spec=103)