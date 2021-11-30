/*
 * Title: +++
 * +++++
 */

// CONSTANTS
const int motorPin = 9;
const int ledPin = 13;

// VARIABLES
bool ledIsHigh = false;


// FUNCTIONS
void flipLED();


void setup()
{
  // put your setup code here, to run once:
  pinMode(motorPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
}

void loop()
{
  // put your main code here, to run repeatedly:
 
  analogWrite(motorPin, 84); // start (max speed)
  flipLED();
  delay(1000);

  analogWrite(motorPin, 184); // middle (stops)
  flipLED();
  delay(1000);

  analogWrite(motorPin, 252); // end (max speed))
  flipLED();
  delay(1000);

  analogWrite(motorPin, 184); // middle (stops)
  flipLED();
  delay(1000);
  
  
}

void flipLED()
{
  ledIsHigh = !ledIsHigh;
  
  if (ledIsHigh)
    digitalWrite(ledPin, HIGH);
  else
    digitalWrite(ledPin, LOW);
}

