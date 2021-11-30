/*
 * Title: +++
 * +++++
 */

// CONSTANTS
const int motorPin = 9;
const int ledPin = 13;

// VARIABLES
signed char motorVal = -90;
bool ledIsHigh = false;


// FUNCTIONS
void flipLED();


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

  motorVal = -motorVal;
  Serial.println((int)motorVal);
  Serial.println((unsigned char)motorVal);
  Serial.println(" ");
 
  analogWrite(motorPin, (unsigned char)motorVal);
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

