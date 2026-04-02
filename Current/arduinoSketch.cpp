const int SAMPLE_PIN = A15;
const int NUM_SAMPLES = 2048;
const int SAMPLE_RATE_HZ = 10000;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;
int samples[NUM_SAMPLES];
void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }
  Serial.println("READY");
}
void loop() {
  if (Serial.available() && Serial.read() == 'G') {
    unsigned long next_sample_time = micros();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      while ((long)(micros() - next_sample_time) < 0) {
      }
      samples[i] = analogRead(SAMPLE_PIN);
      next_sample_time += SAMPLE_INTERVAL_US;
    }
    Serial.println("START");
    for (int i = 0; i < NUM_SAMPLES; i++) {
      Serial.println(samples[i]);
    }
    Serial.println("END");
  }
}