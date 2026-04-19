#include "DHT.h"

// -------------------- PIN DEFINITIONS --------------------
#define DHTPIN 18
#define DHTTYPE DHT11

#define MQ135_PIN 34
#define MQ2_PIN   35

#define TRIG_PIN  25
#define ECHO_PIN  26

#define FLAME_DO  27

// -------------------- OBJECTS --------------------
DHT dht(DHTPIN, DHTTYPE);

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);
  delay(3000);

  dht.begin();

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(FLAME_DO, INPUT);

  Serial.println("timestamp,temperature,humidity,mq135,mq2,flame,distance");
}

// -------------------- LOOP --------------------
void loop() {

  // ---------- DHT11 ----------
  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();

  // ---------- MQ Sensors ----------
  int mq135_value = analogRead(MQ135_PIN);
  int mq2_value   = analogRead(MQ2_PIN);

  // ---------- Flame Sensor ----------
  int flame_state = digitalRead(FLAME_DO); // LOW = flame

  // ---------- Ultrasonic ----------
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  float distance = (duration == 0) ? -1 : duration * 0.034 / 2;

  // ---------- Timestamp ----------
  unsigned long timeStamp = millis();

  // ---------- CSV OUTPUT ----------
  Serial.print(timeStamp);
  Serial.print(",");

  if (isnan(temperature)) Serial.print("0");
  else Serial.print(temperature);
  Serial.print(",");

  if (isnan(humidity)) Serial.print("0");
  else Serial.print(humidity);
  Serial.print(",");

  Serial.print(mq135_value);
  Serial.print(",");

  Serial.print(mq2_value);
  Serial.print(",");

  Serial.print(flame_state == LOW ? 1 : 0);  // 1 = flame
  Serial.print(",");

  Serial.println(distance);

  delay(500);
}