#include <HardwareSerial.h>

HardwareSerial mySerial(2);  // Use UART2

// TX and RX pins (match these with your receiver)
#define RXD2 16  // Not used in transmitter, but must still be declared
#define TXD2 17  // TX pin to send data

void setup() {
  Serial.begin(115200);       // USB Serial for debugging
  mySerial.begin(9600, SERIAL_8N1, RXD2, TXD2);  // Initialize UART2
  Serial.println("UART2 Transmitter Initialized");
}

void loop() {
  // Generate a random ASCII character (printable range)
  char randomChar = random(32, 127);  // ASCII from space to ~

  // Send via UART2
  mySerial.print(randomChar);

  // Debug output
  Serial.print("Sent: ");
  Serial.println(randomChar);

  delay(1000);  // Send one byte per second
}
