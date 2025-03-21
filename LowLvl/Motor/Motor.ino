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

void setup() {
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
}

void loop() {
  setDirection();

}

void setDirection() {

  ledcWrite(pwm_channel, 50);

  digitalWrite(M1IN1, HIGH);
  digitalWrite(M1IN2, LOW);
  digitalWrite(M3IN3, HIGH);
  digitalWrite(M3IN4, LOW);

  digitalWrite(M2IN1, HIGH);
  digitalWrite(M2IN2, LOW);
  digitalWrite(M4IN3, HIGH);
  digitalWrite(M4IN4, LOW);

}