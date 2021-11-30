/* 2019-06-05 */

#include <Servo.h>
#include <Wire.h>
#include <Adafruit_MotorShield.h>


// CONSTANTS
int BUTTON_PIN = 4;
int LIMIT_BOT_PIN = 2;
int LIMIT_TOP_PIN = 3;
int SERVY_PIN = 10;
int STEPPY_PORT = 2;

// SERVO VARIABLES
Servo Servy;  // create servo object to control a servo
//int servy_pos = 0;    // variable to store the servo position

// STEPPER VARIABLES
// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 
// Or, create it with a different I2C address (say for stacking)
// Adafruit_MotorShield AFMS = Adafruit_MotorShield(0x61); 
// Connect a stepper motor with 200 steps per revolution (1.8 degree)
// to motor port #2 (M3 and M4)
Adafruit_StepperMotor *Steppy = AFMS.getStepper(200, STEPPY_PORT);
int STEP_SIZE = 5;  // normally 5

// STATE VARIABLES
int STATE = 0;

// FUNCTION DECLARATIONS
bool isPinHigh(int pin_number);
void setServy(bool hold);
void display_states();

void setup()
{
  // INPUTS SETUP
  pinMode(BUTTON_PIN, INPUT);
  pinMode(LIMIT_BOT_PIN, INPUT_PULLUP);
  pinMode(LIMIT_TOP_PIN, INPUT_PULLUP);
  
  // SERVO SETUP
  Servy.attach(SERVY_PIN);

  // STEPPER SETUP
  AFMS.begin();  // create with the default frequency 1.6KHz
  //AFMS.begin(1000);  // OR with a different frequency, say 1KHz
  Steppy->setSpeed(50);  // rpm (normally 50)

  // SERIAL SETUP
  Serial.begin(9600);
  Serial.println("Hello world!!!");
}

void loop()
{
  // DISPLAY COMPONENT FEEDS
  display_states();
  
  // BLINDS ARE AT THE TOP
  if /**/ (STATE == 0)
  {
    setServy(true);  // hold the servo
    while (!isPinHigh(BUTTON_PIN)) { display_states(); }  // wait until button press
    STATE = 1;
  }

  // BLINDS ARE GOING DOWN
  else if (STATE == 1)
  {
    setServy(false);  // open the servo
    while (isPinHigh(LIMIT_BOT_PIN))  // if limitswitch pressed, pin becomes LOW
    {
      Steppy->step(STEP_SIZE, FORWARD, SINGLE);  // lower blinds until at bottom
      display_states();
    }
    STATE = 2;
  }

  // BLINDS ARE AT THE BOTTOM
  else if (STATE == 2)
  {
    setServy(true);  // hold the servo
    while (!isPinHigh(BUTTON_PIN)) { display_states(); }  // wait until button press
    STATE = 3;
  }

  // BLINDS ARE GOING UP
  else if (STATE == 3)
  {
    setServy(false);  // open the servo
    while (isPinHigh(LIMIT_TOP_PIN))  // if limitswitch pressed, pin becomes LOW
    {
      Steppy->step(STEP_SIZE, BACKWARD, SINGLE);  // raise blinds until at bottom
      display_states();
    }
    STATE = 0;
  }
}


bool isPinHigh(int pin_number)
{
  return (digitalRead(pin_number) == HIGH);
}

void setServy(bool hold)
{
  if (hold) { Servy.write(90); }
  else /**/ { Servy.write(0);  }
}

void display_states()
{
  Serial.println("------------");
  Serial.print("Button Pin: ");
  Serial.print(isPinHigh(BUTTON_PIN));
  Serial.print("\n");
  Serial.print("Limit Top Pin: ");
  Serial.print(isPinHigh(LIMIT_TOP_PIN));
  Serial.print("\n");
  Serial.print("Limit Bot Pin: ");
  Serial.print(isPinHigh(LIMIT_BOT_PIN));
  Serial.print("\n");

  Serial.print("STATE: ");
  Serial.print(STATE);
  Serial.print("\n");  
}

