#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>


int M1ENA = 17;
int M1IN1 = 16;
int M1IN2 = 4;
int M3ENB = 5;
int M3IN3 = 2;
int M3IN4 = 15;



int M2ENA = 26;
int M2IN1 = 13;
int M2IN2 = 12;
int M4ENB = 25;
int M4IN3 = 14;
int M4IN4 = 27;


const int frequency = 500;
const int pwm_channel = 0;
const int resolution = 8;
int speed = 50;


// Service name to the broadcasted to outside world
#define PERIPHERAL_NAME "533's  broadcast"
#define SERVICE_UUID "CD9CFC21-0ECC-42E5-BF22-48AA715CA112"
#define CHARACTERISTIC_INPUT_UUID "66E5FFCE-AA96-4DC9-90C3-C62BBCCD29AC"
#define CHARACTERISTIC_INPUT_UUID_NEW "66E5FFCE-AA96-4DC9-90C3-C62BBCCD29AF"
#define CHARACTERISTIC_OUTPUT_UUID "142F29DD-B1F0-4FA8-8E55-5A2D5F3E2471"

// Output characteristic is used to send the response back to the connected phone
BLECharacteristic *pOutputChar;

// Class defines methods called when a device connects and disconnects from the service
class ServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        Serial.println("BLE Client Connected");
    }
    void onDisconnect(BLEServer* pServer) {
        BLEDevice::startAdvertising();
        Serial.println("BLE Client Disconnected");
    }
};

class InputReceivedCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharWriteState) {
        std::string inputValue = pCharWriteState->getValue();
        if (inputValue.length() > 0) {
          Serial.print("Received Value: ");
          for (int i = 0; i < inputValue.length(); i++) {
            Serial.print(inputValue[i]);
          }
          Serial.println();
          // Send data to client
          delay(1000);
          std::string outputData = "Last received: " + inputValue;
          //pOutputChar->setValue(outputData);
          //pOutputChar->notify();
        }
    }
};

class InputReceivedCallbacksNew: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharWriteState) {
        std::string inputValue = pCharWriteState->getValue();
        if (inputValue.length() > 0) {
          Serial.print("Received Value: ");
          for (int i = 0; i < inputValue.length(); i++) {
            Serial.print(inputValue[i]);
          }
          Serial.println();
          // Send data to client
          delay(1000);
          std::string outputData = "Last received: " + inputValue;
          speed = std::stoi( inputValue );
          //pOutputChar->setValue(outputData);
          //spOutputChar->notify();
        }
    }
};

void setup() {
  Serial.begin(115200);


  pinMode(M1ENA, OUTPUT);
  pinMode(M1IN1, OUTPUT);
  pinMode(M1IN2, OUTPUT); 
  digitalWrite(M1IN1, LOW);
  digitalWrite(M1IN2, LOW);
  pinMode(M3ENB, OUTPUT);
  pinMode(M3IN3, OUTPUT);
  pinMode(M3IN4, OUTPUT); 
  digitalWrite(M3IN3, LOW);
  digitalWrite(M3IN4, LOW);
  
  pinMode(M2ENA, OUTPUT);
  pinMode(M2IN1, OUTPUT);
  pinMode(M2IN2, OUTPUT); 
  digitalWrite(M2IN1, LOW);
  digitalWrite(M2IN2, LOW);
  pinMode(M4ENB, OUTPUT);
  pinMode(M4IN3, OUTPUT);
  pinMode(M4IN4, OUTPUT); 
  digitalWrite(M4IN3, LOW);
  digitalWrite(M4IN4, LOW);

  ledcSetup(pwm_channel, frequency, resolution);
  ledcAttachPin(M1ENA, pwm_channel);
  ledcAttachPin(M3ENB, pwm_channel);

  ledcAttachPin(M2ENA, pwm_channel);
  ledcAttachPin(M4ENB, pwm_channel);


  // Initialize BLE
  BLEDevice::init(PERIPHERAL_NAME);

  // Create the server
  BLEServer *pServer = BLEDevice::createServer();
  
  // Create the service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Handle inputs (sent from app)
  BLECharacteristic *pInputChar 
      = pService->createCharacteristic(
                    CHARACTERISTIC_INPUT_UUID,                                        
                    BLECharacteristic::PROPERTY_WRITE_NR |
                    BLECharacteristic::PROPERTY_WRITE);

  BLECharacteristic *qInputChar 
      = pService->createCharacteristic(
                    CHARACTERISTIC_INPUT_UUID_NEW,                                        
                    BLECharacteristic::PROPERTY_WRITE_NR |
                    BLECharacteristic::PROPERTY_WRITE);

  pOutputChar = pService->createCharacteristic(
                        CHARACTERISTIC_OUTPUT_UUID,
                        BLECharacteristic::PROPERTY_READ |
                        BLECharacteristic::PROPERTY_NOTIFY);

  pServer->setCallbacks(new ServerCallbacks());                  
  pInputChar->setCallbacks(new InputReceivedCallbacks());
  qInputChar->setCallbacks(new InputReceivedCallbacksNew());
  
  // Start service
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();
}

void loop() {
  setDirection();

}

void setDirection() {

  ledcWrite(pwm_channel, speed);

  digitalWrite(M1IN1, HIGH);
  digitalWrite(M1IN2, LOW);
  digitalWrite(M3IN3, HIGH);
  digitalWrite(M3IN4, LOW);

  digitalWrite(M2IN1, HIGH);
  digitalWrite(M2IN2, LOW);
  digitalWrite(M4IN3, HIGH);
  digitalWrite(M4IN4, LOW);

}