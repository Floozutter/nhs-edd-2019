/*
 * Title: +++
 * +++++
 */

const int motorPin = 9;
const int ledPin = 13;

int motorPWM = 0;
int ledIsHigh = false;

void setup()
{
  // put your setup code here, to run once:
  pinMode(motorPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  Serial.begin(9600);
}

void loop()
{
  // put your main code here, to run repeatedly:
  for (int i = 0; i < 255/4; i++)
  {
    motorPWM = i*4;
    ledIsHigh = !ledIsHigh;

    analogWrite(motorPin, motorPWM);
    if (ledIsHigh)
      digitalWrite(ledPin, HIGH);
    else
      digitalWrite(ledPin, LOW);

    Serial.println(motorPWM);
    delay(1000);
  }
}
