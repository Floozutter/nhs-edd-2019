/*
 * Title: +++
 * +++++
 */

#include <Servo.h>

// CONSTANTS
const int motorPin = 9;

// VARIABLES
Servo myservo;


void setup()
{
  // put your setup code here, to run once:
  pinMode(motorPin, OUTPUT);
  myservo.attach(motorPin);
}

void loop()
{
  // put your main code here, to run repeatedly:
  myservo.writeMicroseconds(1000); // full speed clockwise
  delay(1000);

  myservo.writeMicroseconds(1500); // stop
  delay(1000);

  myservo.writeMicroseconds(2000); // full speed counterclockwise
  delay(1000);
  
  myservo.writeMicroseconds(1500); // stop
  delay(1000);
}
