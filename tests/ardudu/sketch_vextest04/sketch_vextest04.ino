/*
 * Title: +++
 * +++++
 */

#include <Servo.h>


// CONSTANTS
const int motorPin = 9;
const int ledPin = 13;

// VARIABLES
Servo vmotor;

// FUNCTIONS
void setVMotor(Servo vex_motor, int power);


void setup()
{
  // put your setup code here, to run once:
  pinMode(motorPin, OUTPUT);
  vmotor.attach(motorPin);

  pinMode(ledPin, OUTPUT);
}

void loop()
{
  // put your main code here, to run repeatedly:

  setVMotor(vmotor, 0);
  delay(1000);
  setVMotor(vmotor, 500);
  delay(1000);
  setVMotor(vmotor, 0);
  delay(1000);
  setVMotor(vmotor, -500);
  delay(1000);
  
}


void setVMotor(Servo vex_motor, int power)
{
  /* Sets the power of a Servo type attached to a VEX Motor Controller.
   * 
   * Args
   *  - vex_motor: the servo attached to the pin that sets the motor speed
   *  - power: a value between -500 and +500
   */
  
  vex_motor.writeMicroseconds(1500 + power);
}
