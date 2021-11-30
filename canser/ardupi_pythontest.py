import serial

def wait_til_phrase(ser, phrase):
    response = ser.read_until()
    print(response)


with serial.Serial("COM6", 9800, timeout=0) as ser:
    msg = ""
    while (msg != "EXIT"):
        msg = input("Enter message: ")
        if msg == "W":
            ser.write(b'W')
            wait_til_phrase(ser)
        elif msg == "L":
            ser.write(b'L')
            wait_til_phrase(ser)
        elif msg == "R":
            ser.write(b'R')
            wait_til_phrase(ser)
    ser.close()
