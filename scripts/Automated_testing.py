#!/usr/bin/sudo python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # for GPU 0,-1 for CPU.
import time
from europilot.Keyboard import LinuxVirtualKeyboard
from europilot.screen import stream_local_game_screen,Box
from europilot.joystickF430 import LinuxVirtualJoystick
import uinput
from pyKey import pressKey, releaseKey, press, sendSequence, showKeys
from UtilScripts import normalize,get_angle
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import _thread
from pynput.keyboard import Key, Listener
import glob
from TestConfigAndUtils import ScnScriptList,CSVlogSave

## Paths and names:
PathToTestModel=r'FinalPilotNets'#must contain /result
HumanDriverName="HumanDriver" #nema for CSV Logs works only with flag RecordHumanDriver=1
## Flags::
ShowSteeringWheel           =1 # shows Steering wheel window                        CAN LOWER REFRESH RATE if on
ShowLaneNetOutput           =0 # shows LaneUnet window                              CAN LOWER REFRESH RATE if on
ShowCaptureScreen           =0 # shows capture screen                               CAN LOWER REFRESH RATE if on

RecordHumanDriver           =0 # this flag will disable NN pilot and alow users to control steering wheel. use for recordin referecnce scenario
ShowTensorflowExecutionTime =0 # shows tensoflow execution time in console         CAN LOWER REFRESH RATE if on
PrintNNOutputAngle          =1 # shows estimated angle in console during execution CAN LOWER REFRESH RATE if on
LOGEveryStep                =1 # logs all NN Pilot outpus and timings of execution

FPS=60


## functions for monitoring of DEL key pressed
def show(key):
    if key == Key.delete:
        return False
## functions for monitoring of DEL key pressed
def input_thread(a_list):
    with Listener(on_press=show) as listener:
        listener.join()            # use input() in Python3
        a_list.append(True)## functions for monitoring of DEL key pressed## functions for monitoring of DEL key pressed## functions for monitoring of DEL key pressed## functions for monitoring of DEL key pressed

#Game controll Sequences
def StartSeq(ScenarioFile):
    #ScenarioFile = 'exec /home/gt/highway/sunny.txt'
    print('resuming game')
    keyboard.emit([uinput.KEY_TAB,uinput.KEY_LEFTSHIFT])
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_F1])
    print("setting up ",ScenarioFile)
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_GRAVE])
    time.sleep(0.5)
    keyboard.typestring(InputString=ScenarioFile)
    time.sleep(1)
    ##press enter
    pressKey('ENTER')
    time.sleep(0.1)
    releaseKey('ENTER')
    #press enter
    time.sleep(2)
    keyboard.emit([uinput.KEY_GRAVE])
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_RIGHTCTRL,uinput.KEY_F9])
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_1])
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_F3])
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_F5])
    time.sleep(0.5)
    print("starting engine")
    keyboard.emit([uinput.KEY_E])
    time.sleep(0.5)
    if not RecordHumanDriver:
        print("setting steering wheel to 0")
        joy.emit(0)
        time.sleep(0.5)
    print("Turn on the head lights")
    keyboard.emit([uinput.KEY_L])
    time.sleep(0.1)
    keyboard.emit([uinput.KEY_L])
    time.sleep(0.1)
    print("disenged parking brake")
    keyboard.emit([uinput.KEY_SPACE])
    time.sleep(0.5)
    print("all set ,starting run!!!")
####Stop sequence#######
def StopSeq():
    print("start braking")
    keyboard.hold([uinput.KEY_DOWN])
    time.sleep(7)
    print("Turn off the head lights")
    keyboard.emit([uinput.KEY_L])
    time.sleep(0.1)
    print("parking brake")
    time.sleep(0.5)
    keyboard.emit([uinput.KEY_SPACE])
    time.sleep(0.5)
    print("stop engine")
    keyboard.emit([uinput.KEY_E])
    print("relese donw button")
    keyboard.relese([uinput.KEY_DOWN])
    print('relese down and press tab+shift keyboard for 7 sec')
    keyboard.relese([uinput.KEY_DOWN])
    keyboard.emit([uinput.KEY_TAB,uinput.KEY_LEFTSHIFT])
def AccAndCCSeq(acctime=5):
    time.sleep(2)
    #make shure that vehicle is in forward gear
    keyboard.emit(uinput.KEY_UP)
    time.sleep(0.001)
    keyboard.emit(uinput.KEY_UP)
    print('pressing up key on keyboard for',str(acctime),'sec')
    keyboard.hold([uinput.KEY_UP])
    time.sleep(acctime)  # change this value for faster velocity
    print("start cruse controll is ON")
    keyboard.emit([uinput.KEY_C])
    time.sleep(0.1)
    print('relese up key on and press down on keyboard for 7 sec')
    keyboard.relese([uinput.KEY_UP])

#Pilot test Function
def testpilot(ModelPath,ScenarioFile,acctime=5,wipers=0):
    print('Starting test on: ',ModelPath)
    if not RecordHumanDriver:
        model = load_model(ModelPath)
        #make initial call to model
        image_data = next(streamer)
        im = Image.fromarray(image_data)
        img_front = im.crop(front_coord)
        arr = image.img_to_array(img_front)
        arr = normalize(arr)
        arr = np.reshape(arr, (1,) + arr.shape)
        angle = get_angle(model(arr))

    IsDelKeyPressed = []
    _thread.start_new_thread(input_thread, (IsDelKeyPressed,))
    print('countdown, Click now on game window!!!:')
    for i in range(5,0,-1):
        print(i)
        time.sleep(1)
    print('Model: ', ModelPath.split('/')[1])
    StartSeq(ScenarioFile=ScenarioFile)
    if wipers:
        keyboard.emit([uinput.KEY_P])
        time.sleep(0.1)
        keyboard.emit([uinput.KEY_P])
        time.sleep(0.1)
        keyboard.emit([uinput.KEY_P])
        time.sleep(0.1)
    AccAndCCSeq(acctime=acctime)
    print("autopilot started press DEL on keyboard to Stop")
    IntTime=time.time()
    log=list()
    if RecordHumanDriver:
        print("recoding human driver")
        while not IsDelKeyPressed:
            time.sleep(0.5)
            print("Human driver, waiting to hit del on keyboard for next scenario")
    else:
        while not IsDelKeyPressed:
            #get image:
            image_data = next(streamer)
            ScreenCaptureTime = time.time()
            im = Image.fromarray(image_data)
            img_front = im.crop(front_coord)
            arr = image.img_to_array(img_front)
            arr = normalize(arr)
            arr = np.reshape(arr, (1,) + arr.shape)

            #estimate engle
            ImageCropTime = time.time()
            #angle = get_angle(model.predict(arr, batch_size=1,verbose = ShowTensorflowExecutionTime))
            angle=get_angle(model(arr))#disabled tensofllow execution time print
            if PrintNNOutputAngle:
                print((angle/(65535/2))*540,'deg')
            #send it to virtual steering wheel
            NNtime= time.time()
            joy.emit(angle)

            #log
            if LOGEveryStep:
                log.append([ScreenCaptureTime-IntTime,ImageCropTime-IntTime,NNtime-IntTime,angle])
    #stop running sequence
    if wipers:
        keyboard.emit([uinput.KEY_P])
    StopSeq()
    return log

if __name__ == "__main__":
    keyboard = LinuxVirtualKeyboard()
    box = Box(10, 47, 1162, 911)
    front_coord = (345, 217, 951, 517)
    joy = LinuxVirtualJoystick()
    streamer = stream_local_game_screen(box=box, default_fps=FPS)
    #input()
    if ShowSteeringWheel:
        os.system("kill -9 $(pgrep -f WheelShow.py)")  # kill steering wheel if running can cause bug
        os.system("python3 scripts/WheelShow.py &")
    if ShowLaneNetOutput:
        os.system("kill -9 $(pgrep -f LaneUnetShow.py)")  # kill steering wheel if running can cause bug
        os.system("python3 scripts/LaneUnetShow.py &")

    if ShowCaptureScreen:
        os.system("kill -9 $(pgrep -f CaptureScreenShow.py)")  # kill steering wheel if running can cause bug
        os.system("python3 scripts/CaptureScreenShow.py &")

    if ShowLaneNetOutput or ShowSteeringWheel:
        print("5 sec to get all external tools to setup")
        time.sleep(5)

    ## read all models to test
    model_path_list = glob.glob(PathToTestModel + "/*.hdf5")

    #model_path_list = [model_path_list[5],model_path_list[17]]#[model_path_list[3],model_path_list[5],model_path_list[10],model_path_list[12]]#model_path_list[14:15]#Selecting specific pilot for debuging and repeating tests!!!!
    model_path_list.sort()
    if RecordHumanDriver:  # overwrite model_path_list and force string in HumanDriverName varible
        newname = 'a/' + HumanDriverName + '.-/'
        model_path_list = [newname]

    # Main Loop:
    for eachscn in ScnScriptList:  # position loop
        for eachacctime in eachscn[1]:  # acc time loop
            for eachmodel in model_path_list:  # each NN model Loop
                log = testpilot(ModelPath=eachmodel, ScenarioFile=eachscn[0], acctime=eachacctime, wipers=eachscn[2])
                csvpath = eachmodel.split('/')[0] + \
                          '/results/' + eachmodel.split('/')[1].split('.')[0].split("-")[0] + \
                          '_' \
                          +eachscn[0].split('/')[3] \
                          + '_' \
                          + eachscn[0].split("/")[-1].split(".")[0] + \
                          "AccTime" + str(eachacctime) + \
                          '_log.csv'
                firstrow = (['Screen Capture Time,', 'Crop Time', 'NN Time', 'Steering wheel angle'])
                if LOGEveryStep:
                    CSVlogSave(data=log, pathtocsv=csvpath, firstrow=firstrow)



