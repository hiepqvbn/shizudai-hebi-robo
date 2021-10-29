#!/usr/bin/env python3

import hebi
from time import sleep, time
from matplotlib import pyplot as plt
import numpy as np

lookup = hebi.Lookup()

# Wait 2 seconds for the module list to populate
sleep(2.0)

family_name = "HEBI"
module_name = "mobileIO"

group = lookup.get_group_from_names([family_name], [module_name])

if group is None:
  print('Group not found: Did you forget to set the module family and name above?')
  exit(1)

# Live Visualization
# Starts logging in the background. Note that logging can be enabled at any time, and that it does not negatively
# affect the performance of your running programs.
group.start_log('dir', 'logs', mkdirs=True)

print('Drag the Sliders and press some buttons on the app screen!')
x_labels = ("1", "2", "3", "4", "5", "6", "7", "8")
x_ticks = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

plt.ion()
f = plt.figure()
plt.ylim([-1, 1])
plt.xticks(x_ticks, x_labels)
plt.xlabel('Digital Inputs and Analog Inputs')
plt.ylabel('[-1 to 1]')
plt.grid(True)

duration = 10.0
start_time = time()
end_time = start_time + duration
current_time = start_time
fbk = hebi.GroupFeedback(group.size)

while current_time < end_time:
  current_time = time()
  fbk = group.get_next_feedback(reuse_fbk=fbk)
  buttons = np.zeros(8)
  sliders = np.zeros(8)
  for i in range(8):
    buttons[i] = fbk.io.b.get_int(i+1)
  for i in range(8):
    if fbk.io.a.has_int(i+1):
      sliders[i] = fbk.io.a.get_int(i+1)
    elif fbk.io.a.has_float(i+1):
      sliders[i] = fbk.io.a.get_float(i+1)
  
  plt.bar(x_ticks, buttons)
  plt.bar(x_ticks, sliders)   
  plt.pause(0.00001)

print('All done!')
