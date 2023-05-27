###NOT WORKING
from europilot.screen import Box
from europilot.train import generate_training_data, Config
#from Automated_testing import StartSeq,StopSeq,AccAndCCSeq,show,input_thread
import threading
import time

# to select joystick file open controllerstate.py and edit line ##['python3.10', '-u', os.path.join(dir_path, 'joystick.py')],#
# set #['python3.10', '-u', os.path.join(dir_path, 'joystickPS3.py')], to used PS3 Controller
class MyConfig(Config):
    # Screen area
    BOX = Box(10, 47, 1162, 911)
    # Screen capture fps
    DEFAULT_FPS = 25


#generate_training_data(config=MyConfig)

#main loop

rec_thread= threading.Thread(target=generate_training_data(config=MyConfig))
rec_thread.start()

while True:
    time.sleep(0.5)
    print("i'm working")
