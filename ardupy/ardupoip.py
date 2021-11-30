import serial


with serial.Serial("COM3", 9800, timeout=1) as ser:
    msg = ""
    while (msg != "EXIT"):
        msg = input("Enter message: ")
        if msg == "H":
            ser.write(b'H')
        elif msg == "L":
            ser.write(b'L')
    ser.close()
