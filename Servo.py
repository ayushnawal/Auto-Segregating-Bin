import RPi.GPIO as GPIO
import time

f = open("transfer.txt","r")
final = f.read()

GPIO.setmode(GPIO.BOARD)

GPIO.setup(12, GPIO.OUT)

p = GPIO.PWM(12, 50)

p.start(7.5)

p.ChangeDutyCycle(7.5)  # turn towards 90 degree

if(final=='0' or final=='3'):
    p.ChangeDutyCycle(2.5)
    time.sleep(3)
    p.ChangeDutyCycle(7.5)
    time.sleep(3)
elif(final=='1' or final=='2' or final=='4' or final=='5'):
    p.ChangeDutyCycle(12.5)
    time.sleep(3)
    p.ChangeDutyCycle(7.5)
    time.sleep(3)

p.stop()
GPIO.cleanup()
