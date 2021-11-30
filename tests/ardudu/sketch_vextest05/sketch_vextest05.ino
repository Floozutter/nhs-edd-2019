/*
 * Title: +++
 * +++++
 */

// CONSTANTS
const int encoderPinA = 3;
const int encoderPinB = 4;

// VARIABLES
int encoderVal = 0;
int encALast = LOW;
int encANow = LOW;


void setup()
{
  // put your setup code here, to run once:
  pinMode(encoderPinA, INPUT);  // new-turn pin
  pinMode(encoderPinB, INPUT);  // direction pin
  Serial.begin(9600);
}

void loop()
{
  // put your main code here, to run repeatedly:

  encANow = digitalRead(encoderPinA);
  if ((encALast == LOW) && (encANow == HIGH))
  {
    if (digitalRead(encoderPinB) == LOW)
      encoderVal--;
    else
      encoderVal++;

    Serial.print("Encoder Value:\t");
    Serial.print(encoderVal);
    Serial.print("\n");
  }
  encALast = encANow;
}
