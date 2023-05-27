import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # for GPU 0,-1 for CPU.
import time
import tkinter as tk
import threading
import numpy as np
from PIL import Image, ImageTk
from europilot.screen import stream_local_game_screen,Box
from tensorflow.keras.models import load_model
try:
    model=load_model(r'../LaneLineLableModels/run_allepoch-8935-val_loss-0.0598-val_acc-0.9794.hdf5')
except:
    model=load_model(r'LaneLineLableModels/run_allepoch-8935-val_loss-0.0598-val_acc-0.9794.hdf5')



root = tk.Tk()



box = Box(10,47,1162,911)
front_coord = (345,217, 951, 517)
streamer = stream_local_game_screen(box=box, default_fps=25)
image_data = next(streamer)
image=Image.fromarray(image_data)
imageint=image.resize((400,400))
imgtoest=np.array(imageint)[None,...]
lanes=np.argmax(model(imgtoest)[0,...],axis=-1)*255
imglanes=Image.fromarray(lanes.astype('int8'),'L')
# Load the image and create a Tkinter PhotoImage object
#imageint = Image.open("steeringwheel.png")
#imageint = imageint.resize((400, 400))
#imageint = imageint.rotate(0)
image = imageint
image4rez = image.crop(front_coord)
originalrez=image4rez.size
#root.geometry("400x400+1164+10")
root.geometry('x'.join(str(originalrez).split(','))[1:-1].replace(' ','')+'+1164+349')
root.title('LaneUnetShow')

tk_image = ImageTk.PhotoImage(image)

# Create a label to display the image
label = tk.Label(root, image=tk_image)
label.pack()
run=0

def update_image(setimage):
    # Rotate the image based on the axis value
    #image = imageint
    #image = image.resize((300, 300))
    #image = image.rotate(-axis * 540)
    #print(axis)
    tk_image = ImageTk.PhotoImage(setimage)
    label.configure(image=tk_image)
    label.image = tk_image
    root.update_idletasks()  # This line will update the GUI




def event_loop():
    #update_image(setimage=image)
    while True:
        #newframe=next(streamer)
        #image=Image.fromarray(newframe)
        #image=image.resize((400,400))
        image_data = next(streamer)
        image = Image.fromarray(image_data)
        imageint = image.resize((400, 400))
        imgtoest = np.array(imageint)[None, ...]
        lanes = np.argmax(model(imgtoest)[0, ...], axis=-1) * 255
        imglanes = Image.fromarray(lanes.astype('int8'), 'L')
        imglanes=imglanes.resize(originalrez)
        update_image(setimage=imglanes)
    on_closing()



event_thread = threading.Thread(target=event_loop)
event_thread.start()

# Start the event loop
#event_loop()
root.mainloop()
