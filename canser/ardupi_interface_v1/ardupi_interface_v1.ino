/*
 * Title: +++
 * +++++
 */

#include <Servo.h>


// PIN CONSTANTS
// Integer constants for pin numbers.

// LED pins (write)
const int ledPinInfo = 13;
const int ledPinStop = 10;

// motor pins (write)
const int motorPin = 9;

// rotary encoder pins (read)
const int encoderPinA = 3;
const int encoderPinB = 4;


// CONFIG CONSTANTS
const int DEFAULT_MOTORPOWER = 350;
const int TRACK_ORIENTATION = 1;
// If TRACK_ORIENTATION is +1, positive motor power would increase encoder value.
// If TRACK_ORIENTATION is -1, positive motor power would decrease encoder value.
// Through the use of this constant, positive motor power ALWAYS increases encoder value.

const int ENCODERPOS_HOME = 0;
const int ENCODERPOS_LEFT = +100;
const int ENCODERPOS_RIGHT = -100;


// GLOBAL VARIABLES
Servo vexMotor;
int encoderValue;


// FUNCTION DECLARATIONS
void setVMotor(Servo vexMotor, int power);
int updateEncoderValue(int old_encodervalue);
void moveTrack(Servo VMotor, int pos);
int awaitSerialByte();


// SETUP
void setup()
{
  // PIN SETUP
  pinMode(ledPinInfo, OUTPUT);
  pinMode(ledPinStop, OUTPUT);
  pinMode(motorPin, OUTPUT);
  pinMode(encoderPinA, INPUT);
  pinMode(encoderPinB, INPUT);
  
  // OBJECT SETUP
  vexMotor.attach(motorPin);
  
  // COMMUNICATION SETUP
  Serial.begin(9600);

  // wait for Python communication in serial
  bool is_python_there = false;
  int serialbyte;
  while (!is_python_there)
  {
    serialbyte = awaitSerialByte();
    // capital W indicates "hello World" (comms start)
    if (serialbyte == 'W')  // wait for a hello from Python
    {
      Serial.println('w');  // signal to Python that a hello has been received
      is_python_there = true;  // break out of while loop
    }  // end if
  }  // end while
  encoderValue = ENCODERPOS_HOME;  // set current encoder value as home
}  // end setup


// MAIN LOOP
void loop()
{
  // wait for communication from Python
  int serialbyte;
  serialbyte = awaitSerialByte();
  
  // choose action based on Python communication
  if ((serialbyte == 'L') || (serialbyte == 'R'))  // track move action
  {
    int encoder_destination;
    // set the correct destination depending on whether 'L' or 'R'
    if /**/ (serialbyte == 'L') { encoder_destination = ENCODERPOS_LEFT; }
    else if (serialbyte == 'R') { encoder_destination = ENCODERPOS_RIGHT; }

    // move the track to eject the contents of the bin
    moveTrack(vexMotor, encoder_destination);
    delay(1000);
    moveTrack(vexMotor, ENCODERPOS_HOME);
    delay(1000);

    // communicate to Python that the action is complete
    if /**/ (serialbyte == 'L') { Serial.println('l'); }
    else if (serialbyte == 'R') { Serial.println('r'); }
  }
}


// FUNCTION DEFINITIONS
// User defined functions! How fun.

void setVMotor(Servo VMotor, int power)
{
  /* Sets the power of a Servo type attached to a VEX Motor Controller.
   * 
   * ARGS
   *  - vex_motor: the servo attached to the pin that sets the motor speed
   *  - power: a value between -500 and +500
   */
  int valid_power = power;  // modified power value to guarantee a valid PWM
  if /**/ (power > +500) { valid_power = +500; }
  else if (power < -500) { valid_power = -500; }
  VMotor.writeMicroseconds(1500 + valid_power);
}

int updateEncoderValue(int old_encodervalue)
{
  /* Returns a new encoder value by polling the VEX Shaft Encoder.
   * 
   * ARGS
   *  - old_encodervalue: The current encoder value before the new poll.
   */
  static int encALast = LOW;
  static int encANow = LOW;
  int new_encodervalue = old_encodervalue;  // difference between old and new encoder values
  encANow = digitalRead(encoderPinA);  // PERSISTENT CHANGE
  // encoder delta occurs if reading changed from LOW to HIGH
  if ((encALast == LOW) && (encANow == HIGH))
  {
    if (digitalRead(encoderPinB) == LOW) { new_encodervalue--; }
    else /* ------------------------- */ { new_encodervalue++; }
  }
  encALast = encANow;  // PERSISTENT CHANGE
  // return the new encoder value
  Serial.print("e"); Serial.print(new_encodervalue); Serial.print("\n");
  return new_encodervalue;
}

void moveTrack(Servo VMotor, int pos)
{
  /* Moves the track by powering the motor until...
   * The inputted encoder position is reached.
   * 
   * ARGS
   *  - pos: the newly desired encoder position
   */
  bool reached = false;
  int motor_dir = +1;  // current motor direction (+ direction -> + encoding)
  while (!reached)
  {
    encoderValue = updateEncoderValue(encoderValue);  // global change
    if /**/ (encoderValue < pos) { motor_dir = +1; }
    else if (encoderValue > pos) { motor_dir = -1; }
    else /* ----------------- */ { motor_dir = 0; reached = true; }
    setVMotor(VMotor, (TRACK_ORIENTATION * motor_dir) * DEFAULT_MOTORPOWER);
  }
  setVMotor(VMotor, 0);  // turn off motor
}

int awaitSerialByte()
{
  int serialbyte;  // oldest byte in serial buffer (function output)
  bool wait = true;  // control variable for maintaining loop
  while (wait)
  {
    // see if there's incoming serial data:
    if (Serial.available() > 0)
    {
      // read the oldest byte in the serial buffer:
      serialbyte = Serial.read();
      wait = false;  // break out of loop
    }
  }  // end while
  return serialbyte;
}

