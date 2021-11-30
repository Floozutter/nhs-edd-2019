import serial


with serial.Serial("COM6", 9800, timeout=1) as ser:
    msg = ""
    while (msg != "EXIT"):
        msg = input("Enter message: ")
        if msg == "W":
            ser.write(b'W')
        elif msg == "L":
            ser.write(b'L')
        elif msg == "R":
            ser.write(b'R')
    ser.close()
