import serial
import time
import random




on = True
with serial.Serial("COM3", 9800, timeout=1) as ser:
    for i in range(10000):
        time.sleep(float(random.randint(10, 500))/1000.0)
        on = not on
        if on:
            print("...On!")
            ser.write(b'H')
        else:
            print("...Off!")
            ser.write(b'L')
    ser.close()
