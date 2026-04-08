#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <MPU6050.h>
#include <math.h>

// -------- OBJECTS --------
Adafruit_ADS1115 ads;
MAX30105 particleSensor;
MPU6050 mpu;

// -------- SETTINGS --------
const int SAMPLE_RATE = 50;
const int WINDOW_SIZE = 50;

// -------- BUFFERS --------
float irBuffer[WINDOW_SIZE];
float dispBuffer[WINDOW_SIZE];
float axBuffer[WINDOW_SIZE];

int bufferIndex = 0;

// -------- VARIABLES --------
unsigned long lastBeat = 0;
float bpm = 0;

// -------- SMOOTHING --------
float smooth(float val) {
  static float prev = 0;
  float out = 0.8 * prev + 0.2 * val;
  prev = out;
  return out;
}

// -------- UTIL FUNCTIONS --------

float computeAmplitude(float *arr) {
  float minV = arr[0], maxV = arr[0];
  for (int i = 1; i < WINDOW_SIZE; i++) {
    if (arr[i] < minV) minV = arr[i];
    if (arr[i] > maxV) maxV = arr[i];
  }
  return maxV - minV;
}

float computePower(float *arr) {
  float sum = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float norm = arr[i] / 50000.0;
    sum += norm * norm;
  }
  return sum / WINDOW_SIZE;
}

// -------- SETUP --------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(21, 22);

  ads.begin();
  particleSensor.begin(Wire);
  particleSensor.setup();

  mpu.initialize();

  Serial.println("cardiac_amplitude,cardiac_frequency,respiratory_amplitude,slow_wave_power,cardiac_power,mean_arterial_pressure");
}

// -------- LOOP --------

void loop() {

  // -------- READ BPW (ADS1115) --------
  int16_t q1 = ads.readADC_SingleEnded(0);
  int16_t q2 = ads.readADC_SingleEnded(1);
  int16_t q3 = ads.readADC_SingleEnded(2);
  int16_t q4 = ads.readADC_SingleEnded(3);

  int x = (q1 + q4) - (q2 + q3);
  int y = (q1 + q2) - (q3 + q4);

  float displacement = sqrt((float)(x*x + y*y));
  displacement = smooth(displacement);

  // -------- READ PPG --------
  long ir = particleSensor.getIR();

  // -------- READ MPU --------
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  // -------- STORE --------
  irBuffer[bufferIndex] = ir;
  dispBuffer[bufferIndex] = displacement;
  axBuffer[bufferIndex] = ax;

  // -------- HEART RATE --------
  if (checkForBeat(ir)) {
    unsigned long delta = millis() - lastBeat;
    lastBeat = millis();

    if (delta > 0) {
      float newBpm = 60.0 / (delta / 1000.0);
      bpm = 0.9 * bpm + 0.1 * newBpm; // smoothing
    }
  }

  // Finger detection
  if (ir < 50000) bpm = 0;

  bufferIndex++;

  // -------- PROCESS WINDOW --------
  if (bufferIndex >= WINDOW_SIZE) {

    float cardiac_amp_raw = computeAmplitude(irBuffer);
    float resp_amp_raw = computeAmplitude(dispBuffer);
    float slow_power_raw = computePower(dispBuffer);
    float cardiac_power_raw = computePower(irBuffer);

    // 🔥 -------- NORMALIZATION (CRITICAL) --------

    float cardiac_amp = map(cardiac_amp_raw, 0, 20000, 25, 45);
    float cardiac_freq = constrain(bpm / 60.0, 0.8, 2.2);
    float resp_amp = map(resp_amp_raw, 0, 5000, 2, 15);
    float slow_power = constrain(slow_power_raw, 0.8, 1.0);
    float cardiac_power = constrain(cardiac_power_raw / 100.0, 0.001, 0.05);
    float map_estimate = 90; // constant (demo)

    // -------- OUTPUT --------
    Serial.print(cardiac_amp); Serial.print(",");
    Serial.print(cardiac_freq); Serial.print(",");
    Serial.print(resp_amp); Serial.print(",");
    Serial.print(slow_power); Serial.print(",");
    Serial.print(cardiac_power); Serial.print(",");
    Serial.println(map_estimate);

    bufferIndex = 0;
  }

  delay(1000 / SAMPLE_RATE);
}