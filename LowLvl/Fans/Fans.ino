const int pin = 2;           // Pin 2
const unsigned long interval = 30000; // 2 minutes in milliseconds

unsigned long previousMillis = 0;
bool pinState = LOW;

void setup() {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, pinState); // Start with pin OFF

  Serial.begin(9600);          // Start Serial Monitor
  Serial.println("Starting...");
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Toggle the pin state
    pinState = !pinState;
    digitalWrite(pin, pinState);

    // Print to Serial Monitor
    if (pinState == HIGH) {
      Serial.println("Pin 2 turned ON");
    } else {
      Serial.println("Pin 2 turned OFF");
    }
  }
}
