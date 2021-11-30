import serial


with serial.Serial("COM3", 9800, timeout=1) as ser:
    while True:
        print(ser.read_until().decode("utf-8"), end='')
