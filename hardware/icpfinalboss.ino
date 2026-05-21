/*
  icpfinalboss.ino
  ================
  ESP32 firmware for non-invasive ICP monitoring.
  Outputs raw sample CSV at exactly 50 Hz via hardware timer interrupt.
  Python pipeline does all feature extraction — no onboard math.

  Columns: timestamp_ms,ir_raw,red_raw,disp_raw,disp_x,disp_y,
           ax,ay,az,gx,gy,gz,artifact_flag,session_label

  session_label: 0=normal, 3=valsalva, 4=recovery
  Valsalva button on GPIO 15: press to start (label→3), press again to stop (label→4)
*/

#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include "MAX30105.h"
#include <MPU6050.h>

// ── Hardware objects ──────────────────────────────────────────────────────────
Adafruit_ADS1115 ads;
MAX30105 particleSensor;
MPU6050 mpu;

// ── Timer ─────────────────────────────────────────────────────────────────────
hw_timer_t *timer = NULL;
volatile bool sample_flag = false;
void IRAM_ATTR onTimer() { sample_flag = true; }

// ── Valsalva button ───────────────────────────────────────────────────────────
#define VALSALVA_BTN 15
volatile uint8_t session_label = 0;   // 0=normal, 3=valsalva, 4=recovery
unsigned long last_btn_ms = 0;

// ── Artifact detection ────────────────────────────────────────────────────────
// Ring buffer for z-score spike detection on displacement
#define ART_BUF 50
float disp_hist[ART_BUF];
int disp_idx = 0;
bool art_buf_full = false;

uint8_t detect_artifact(float disp_val, int16_t ax, int16_t ay, int16_t az) {
  // IMU motion artifact: acceleration magnitude > 1.5g (~15000 raw)
  float accel_mag = sqrt((float)(ax*ax + ay*ay + az*az));
  if (accel_mag > 20000) return 1;

  // Displacement z-score > 3.5 sigma
  if (art_buf_full) {
    float sum = 0, sum2 = 0;
    for (int i = 0; i < ART_BUF; i++) { sum += disp_hist[i]; sum2 += disp_hist[i]*disp_hist[i]; }
    float mean = sum / ART_BUF;
    float std  = sqrt(sum2 / ART_BUF - mean * mean);
    if (std > 0 && fabs(disp_val - mean) > 3.5 * std) return 1;
  }

  disp_hist[disp_idx] = disp_val;
  disp_idx = (disp_idx + 1) % ART_BUF;
  if (disp_idx == 0) art_buf_full = true;
  return 0;
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  ads.begin();
  ads.setDataRate(RATE_ADS1115_860SPS);  // 860 SPS → ~1.16ms/ch × 4 = ~4.6ms; fits in 20ms (50 Hz) budget
  particleSensor.begin(Wire);
  particleSensor.setup();
  mpu.initialize();

  pinMode(VALSALVA_BTN, INPUT_PULLUP);

  // Hardware timer: prescaler=80 → 1 MHz clock; alarm=20000 → 50 Hz
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 20000, true);
  timerAlarmEnable(timer);

  Serial.println("timestamp_ms,ir_raw,red_raw,disp_raw,disp_x,disp_y,ax,ay,az,gx,gy,gz,artifact_flag,session_label");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  // Valsalva button debounce (300 ms)
  if (digitalRead(VALSALVA_BTN) == LOW && millis() - last_btn_ms > 300) {
    session_label = (session_label == 3) ? 4 : 3;
    last_btn_ms = millis();
  }

  if (!sample_flag) return;
  sample_flag = false;

  unsigned long ts = millis();

  // ADS1115 quad-photodiode (BPW34)
  int16_t q1 = ads.readADC_SingleEnded(0);
  int16_t q2 = ads.readADC_SingleEnded(1);
  int16_t q3 = ads.readADC_SingleEnded(2);
  int16_t q4 = ads.readADC_SingleEnded(3);

  int16_t disp_x   = (q1 + q4) - (q2 + q3);
  int16_t disp_y   = (q1 + q2) - (q3 + q4);
  float   disp_raw = sqrt((float)(disp_x*disp_x + disp_y*disp_y));

  // MAX30102 PPG
  long ir_raw  = particleSensor.getIR();
  long red_raw = particleSensor.getRed();

  // MPU6050
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  uint8_t artifact = detect_artifact(disp_raw, ax, ay, az);

  // Output raw CSV row
  Serial.print(ts);       Serial.print(",");
  Serial.print(ir_raw);   Serial.print(",");
  Serial.print(red_raw);  Serial.print(",");
  Serial.print(disp_raw); Serial.print(",");
  Serial.print(disp_x);   Serial.print(",");
  Serial.print(disp_y);   Serial.print(",");
  Serial.print(ax);       Serial.print(",");
  Serial.print(ay);       Serial.print(",");
  Serial.print(az);       Serial.print(",");
  Serial.print(gx);       Serial.print(",");
  Serial.print(gy);       Serial.print(",");
  Serial.print(gz);       Serial.print(",");
  Serial.print(artifact); Serial.print(",");
  Serial.println(session_label);
}
