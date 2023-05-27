from europilot.joystickF430 import LinuxVirtualJoystick
joy = LinuxVirtualJoystick()
import numpy as np
import time
while True:
    for i in range(-200,200,5):
        ii=int(i*65535/(540*2))
        joy.emit(ii)
        time.sleep(0.2)