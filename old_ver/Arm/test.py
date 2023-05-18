# #!/usr/bin/env python3

# import hebi
# from math import pi, sin
# from time import sleep, time

# lookup = hebi.Lookup()

# # Wait 2 seconds for the module list to populate
# sleep(2.0)

# family_name = "Arm"
# module_name = "J3_elbow"

# group = lookup.get_group_from_names([family_name], [module_name])

# if group is None:
#   print('Group not found: Did you forget to set the module family and name above?')
#   exit(1)

# group_command  = hebi.GroupCommand(group.size)
# group_feedback = hebi.GroupFeedback(group.size)

# # Start logging in the background
# group.start_log('logs', mkdirs=True)

# freq_hz = 0.5                 # [Hz]
# freq    = freq_hz * 2.0 * pi  # [rad / sec]
# amp     = pi /10           # [rad] (45 degrees)

# duration = 2              # [sec]
# start = time()
# t = time() - start

# while t < duration:
#   # Even though we don't use the feedback, getting feedback conveniently
#   # limits the loop rate to the feedback frequency
#   group.get_next_feedback(reuse_fbk=group_feedback)
#   t = time() - start

#   group_command.position = amp * sin(freq * t)
#   group.send_command(group_command)

# pos = amp*sin(freq*t)

# while t<10:
#     group.get_next_feedback(reuse_fbk=group_feedback)
#     t = time() - start

#     group_command.position = pos
#     group.send_command(group_command)


# # Stop logging. `log_file` contains the contents of the file
# log_file = group.stop_log()
# hebi.util.plot_logs(log_file, 'position')
import hebi
from time import sleep

lookup = hebi.Lookup()

# Wait 2 seconds for the module list to populate, and then print out its contents
sleep(2.0)

for entry in lookup.entrylist:
  print(entry)

print('NOTE:')
print('  The listing above should show the information for all the modules')
print('  on the local network. If this is empty make sure that the modules')
print('  are connected, powered on, and that the status LEDs are displaying')
print('  a green soft-fade.')