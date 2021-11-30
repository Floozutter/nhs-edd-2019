/*
 * Title: +++
 * +++++
 */

#include <Servo.h>

// CONSTANTS
const int motorPin = 9;
const int ledPin = 13;

// VARIABLES
Servo myservo;

void setup()
{
  // put your setup code here, to run once:
  pinMode(motorPin, OUTPUT);
  myservo.attach(motorPin);

  pinMode(ledPin, OUTPUT);
}

void loop()
{
  // put your main code here, to run repeatedly:

  digitalWrite(ledPin, HIGH);
  for (int i = 1000; i < 2001; i++)
  {
    myservo.writeMicroseconds(i);
    delay(5);
  }

  for (int i = 2000; i > 999; i--)
  {
    myservo.writeMicroseconds(i);
    delay(5);
  }
  digitalWrite(ledPin, LOW);
  
}
