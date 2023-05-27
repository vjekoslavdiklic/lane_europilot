import time
import threading
import pygame
import tkinter as tk
from PIL import Image, ImageTk


root = tk.Tk()
root.geometry("187x187+1164+687")
root.title('Steering Wheel Position')


pygame.init()
pygame.joystick.init()
pygame.joystick.get_count()
try:
    joystick = pygame.joystick.Joystick(0)
except:
    a=1
    while a:
        event = pygame.event.wait()
        if event.type == pygame.JOYDEVICEADDED:
            joystick = pygame.joystick.Joystick(0)
            a=0
joystick.init()

# Load the image and create a Tkinter PhotoImage object
imageint = Image.open("steeringwheel.png")
imageint = imageint.resize((187, 187))
imageint = imageint.rotate(0)
image = imageint
tk_image = ImageTk.PhotoImage(image)

# Create a label to display the image
label = tk.Label(root, image=tk_image)
label.pack()


def update_image(axis):
    # Rotate the image based on the axis value
    image = imageint
    image = image.resize((187, 187))
    image = image.rotate(-axis * 540)
    #print(axis)
    tk_image = ImageTk.PhotoImage(image)
    label.configure(image=tk_image)
    label.image = tk_image
    root.update_idletasks()  # This line will update the GUI


def event_loop():
    update_image(0)
    while True:
        event = pygame.event.wait()
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                axis = joystick.get_axis(event.axis)
                update_image(axis)
        if event.type == pygame.JOYDEVICEREMOVED:
            pygame.joystick.quit()
            exit()

event_thread = threading.Thread(target=event_loop)
event_thread.start()
# Start the event loop
#event_loop()
root.mainloop()
