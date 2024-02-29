#include <PDM.h>

short  sampleBuffer[256];
volatile int samplesRead;

const int BUTTON_TRIGGER = 10;
const int SIGNAL_OUTPUT = 5;

int SIZE_PER_SAMPLE = 4000; // 4000 per second
int size2Send = 0;
bool ready = true;

void setup() {
  // init output
  Serial.begin(9600);
  while (!Serial);

  // init sampling triger
  pinMode(BUTTON_TRIGGER, INPUT);
  pinMode(SIGNAL_OUTPUT, OUTPUT);
  digitalWrite(SIGNAL_OUTPUT, HIGH);
  // init LED as signal
  pinMode(LED_BUILTIN, OUTPUT);

  // configure the data receive callback
  PDM.onReceive(onPDMdata);

  // init PDM
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
}

void loop() {
  if (size2Send == 0) {
    if (ready && digitalRead(BUTTON_TRIGGER) == HIGH) {
      // send transimision starting signal
      ready = false;
      Serial.print("Button Pressed, start sending ");
      Serial.print(SIZE_PER_SAMPLE);
      Serial.println(" samples:");
      // reset size of samples to send
      size2Send = SIZE_PER_SAMPLE;
      // transmission start, turn on the LED
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (!ready && digitalRead(BUTTON_TRIGGER) == LOW) {
      // transmission finished, turn off the LED, ready for next transmission
      digitalWrite(LED_BUILTIN, LOW);
      ready = true;
      delay(100);
    }
  }


  // wait for samples to be read
  if (samplesRead > 0) {
    for (int i = 0; i < samplesRead; ++i) {
      // output samples of certain size
      if (size2Send == 0) break;
      Serial.println(sampleBuffer[i]);
      size2Send--;
    }
  }

  // clear the read count 
  samplesRead = 0;
}

void onPDMdata() {
  // query the number of bytes available
  int bytesAvailable = PDM.available();

  // read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample
  samplesRead = bytesAvailable / 2;
}
