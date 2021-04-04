from picamera import PiCamera
from time import sleep

camera = PiCamera()

sleep(2) #Allows brightness adjustment, makes camera quality better
camera.capture('/home/pi/Documents/main/test.png')