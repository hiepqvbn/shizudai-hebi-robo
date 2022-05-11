import imp
from xbox360controller import Xbox360Controller
import time

# with Xbox360Controller() as controller:
#     controller.set_rumble(0.5, 0.5, 1000)
#     time.sleep(6.5)

print(Xbox360Controller().has_led)

with Xbox360Controller() as controller:
    controller.set_led(Xbox360Controller.LED_ROTATE)
    time.sleep(1)
    controller.set_led(Xbox360Controller.LED_OFF)