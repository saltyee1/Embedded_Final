# main.py
# Some ports need to import 'sleep' from 'time' module
from math import sqrt
from machine import sleep, SoftI2C, Pin, UART, ADC, Timer
from utime import ticks_diff, ticks_us, ticks_ms

from max30102 import MAX30102, MAX30105_PULSE_AMP_MEDIUM

uart = UART(1,baudrate=9600, tx=17, rx=16) #以UART傳輸至python介面

AD8232 = ADC(Pin(2))   #設定15號腳位為ADC
AD8232.atten (ADC.ATTN_11DB) #設定11dB衰減，輸人電壓上限3.3V。
AD8232.width(ADC.WIDTH_12BIT) #設定成12bit解析度



# Initialize variables
avered = 0.0
aveir = 0.0
sumirrms = 0.0
sumredrms = 0.0
SpO2 = 0.0
ESpO2 = 90.0                # Initial value
FSpO2 = 0.7                 # SpO2 estimation parameter
frate = 0.95                # Low-pass filter parameter to eliminate noise
i = 0
Num = 30                    # Calculate once every 30 samples

FINGER_ON = 7000            # Minimum IR value (to determine if the finger is on the sensor)
MINIMUM_SPO2 = 90.0         # Minimum SpO2 value

Spo2_re = [0.0] * 30        # Array to store 30 samples
Spo2_ave = 0.0
num = 0


class HeartRateMonitor:
    """A simple heart rate monitor that uses a moving window to smooth the signal and find peaks."""

    def __init__(self, sample_rate=100, window_size=20, smoothing_window=10):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.samples = []
        self.timestamps = []
        self.filtered_samples = []

    def add_sample(self, sample):
        """Add a new sample to the monitor."""
        timestamp = ticks_ms()
        self.samples.append(sample)
        self.timestamps.append(timestamp)

        # Apply smoothing
        if len(self.samples) >= self.smoothing_window:
            smoothed_sample = (
                sum(self.samples[-self.smoothing_window :]) / self.smoothing_window
            )
            self.filtered_samples.append(smoothed_sample)
        else:
            self.filtered_samples.append(sample)

        # Maintain the size of samples and timestamps
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
            self.timestamps.pop(0)
            self.filtered_samples.pop(0)

    def find_peaks(self):
        """Find peaks in the filtered samples."""
        peaks = []

        if len(self.filtered_samples) < 3:  # Need at least three samples to find a peak
            return peaks

        # Calculate dynamic threshold based on the min and max of the recent window of filtered samples
        recent_samples = self.filtered_samples[-self.window_size :]
        min_val = min(recent_samples)
        max_val = max(recent_samples)
        threshold = (
            min_val + (max_val - min_val) * 0.5
        )  # 50% between min and max as a threshold

        for i in range(1, len(self.filtered_samples) - 1):
            if (
                self.filtered_samples[i] > threshold
                and self.filtered_samples[i - 1] < self.filtered_samples[i]
                and self.filtered_samples[i] > self.filtered_samples[i + 1]
            ):
                peak_time = self.timestamps[i]
                peaks.append((peak_time, self.filtered_samples[i]))

        return peaks

    def calculate_heart_rate(self):
        """Calculate the heart rate in beats per minute (BPM)."""
        peaks = self.find_peaks()

        if len(peaks) < 3:
            return None  # Not enough peaks to calculate heart rate

        # Calculate the average interval between peaks in milliseconds
        intervals = []
        for i in range(1, len(peaks)):
            interval = ticks_diff(peaks[i][0], peaks[i - 1][0])
            intervals.append(interval)

        average_interval = sum(intervals) / len(intervals)

        # Convert intervals to heart rate in beats per minute (BPM)
        heart_rate = (
            60000 / average_interval
        )  # 60 seconds per minute * 1000 ms per second

        return heart_rate
    
def AD8232_sampling(self):
    AD8232_val = (AD8232.read() / 4095 * 3.3)  # 讀取ADC的12bit資料，並轉換成實際電壓值
    formatted_value = "{:.2f}".format(AD8232_val)  # 格式化為兩位小數
    uart_string = "0\t0\t" + formatted_value + "\n"  # 構建最終字串
    #print(uart_string)
    uart.write(uart_string)  # 傳輸字串數值
    
tim1=Timer(1)
tim1.init(period = 50, mode = Timer.PERIODIC, callback = AD8232_sampling)

last_heartrate = 80
def main():
    global avered, aveir, sumirrms, sumredrms, SpO2, ESpO2, i, num, Spo2_ave, last_heartrate
    # I2C software instance
    i2c = SoftI2C(
        sda=Pin(21),  # Here, use your I2C SDA pin
        scl=Pin(22),  # Here, use your I2C SCL pin
        freq=400000,
    )  # Fast: 400kHz, slow: 100kHz

    # Examples of working I2C configurations:
    # Board             |   SDA pin  |   SCL pin
    # ------------------------------------------
    # ESP32 D1 Mini     |   22       |   21
    # TinyPico ESP32    |   21       |   22
    # Raspberry Pi Pico |   16       |   17
    # TinyS3			|	 8		 |    9

    # Sensor instance
    sensor = MAX30102(i2c=i2c)  # An I2C instance is required

    # Scan I2C bus to ensure that the sensor is connected
    if sensor.i2c_address not in i2c.scan():
        print("Sensor not found.")
        return
    elif not (sensor.check_part_id()):
        # Check that the targeted sensor is compatible
        print("I2C device ID not corresponding to MAX30102 or MAX30105.")
        return
    else:
        print("Sensor connected and recognized.")

    # Load the default configuration
    print("Setting up sensor with default configuration.", "\n")
    sensor.setup_sensor()

    # Set the sample rate to 400: 400 samples/s are collected by the sensor
    sensor_sample_rate = 400
    sensor.set_sample_rate(sensor_sample_rate)

    # Set the number of samples to be averaged per each reading
    sensor_fifo_average = 8
    sensor.set_fifo_average(sensor_fifo_average)

    # Set LED brightness to a medium value
    sensor.set_active_leds_amplitude(MAX30105_PULSE_AMP_MEDIUM)

    # Expected acquisition rate: 400 Hz / 8 = 50 Hz
    actual_acquisition_rate = int(sensor_sample_rate / sensor_fifo_average)

    sleep(1)

    print(
        "Starting data acquisition from RED & IR registers...",
        "press Ctrl+C to stop.",
        "\n",
    )
    sleep(1)

    # Initialize the heart rate monitor
    hr_monitor = HeartRateMonitor(
        # Select a sample rate that matches the sensor's acquisition rate
        sample_rate=actual_acquisition_rate,
        # Select a significant window size to calculate the heart rate (2-5 seconds)
        window_size=int(actual_acquisition_rate * 3),
    )

    # Setup to calculate the heart rate every 2 seconds
    hr_compute_interval = 5  # seconds
    ref_time = ticks_ms()  # Reference time

    while True:
        # The check() method has to be continuously polled, to check if
        # there are new readings into the sensor's FIFO queue. When new
        # readings are available, this function will put them into the storage.
        sensor.check()

        # Check if the storage contains available samples
        if sensor.available():
            # Access the storage FIFO and gather the readings (integers)
            red_reading = sensor.pop_red_from_storage()
            ir_reading = sensor.pop_ir_from_storage()
            ###################################
            i += 1
            red = red_reading
            ir = ir_reading
            fred = float(red)  # Convert to double
            fir = float(ir)  # Convert to double
            avered = avered * frate + fred * (1.0 - frate)  # Average red value using low-pass filter
            aveir = aveir * frate + fir * (1.0 - frate)  # Average IR value using low-pass filter
            sumredrms += (fred - avered) ** 2  # Sum of squared AC components of red value
            sumirrms += (fir - aveir) ** 2  # Sum of squared AC components of IR value

            if (i % Num) == 0:
                R = (sqrt(sumredrms) / avered) / (sqrt(sumirrms) / aveir)
                SpO2 = -23.3 * (R - 0.4) + 100
                ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2  # Low-pass filter

                if ESpO2 <= MINIMUM_SPO2:
                    ESpO2 = MINIMUM_SPO2  # Limit minimum value

                if ESpO2 > 100:
                    ESpO2 = 99.9  # Limit maximum value

                sumredrms = 0.0
                sumirrms = 0.0
                SpO2 = 0
                i = 0

                Spo2_re[num] = ESpO2  # Store the result every 30 samples
                print(".", end="")  # Print a dot

                num += 1

                if num == 10:  # After 30 samples
                    print("OK!")
                    for ii in range(5, 10):  # Average from the 16th to the 30th sample
                        Spo2_ave += Spo2_re[ii]  # Collect sample values

                    Spo2_ave /= 5
                    Spo2_ave = round(Spo2_ave, 2)
                    
                    if heart_rate != None :
                        print("Heart Rate: {:.0f} BPM".format(heart_rate))
                        print("SPO2 (%) =", Spo2_ave)
                        uart_string = "{:.2f}\t{:.0f}\t{}\n".format(Spo2_ave, heart_rate, 0)  # 構建最終字串
                        uart.write(uart_string)  # 傳輸字串數值
                    #if (heart_rate < last_heartrate + 10 or heart_rate > last_heartrate - 10) :
                        # = "{:.0f}\t{:.0f}\t{}\n".format(Spo2_ave, heart_rate, 0)  # 構建最終字串
                    #else :
                    #    uart_string = "{:.0f}\t{:.0f}\t{}\n".format(Spo2_ave, 0, 0)
                    #print(uart_string)
                    last_heartrate = heart_rate
                    avered = 0.0
                    aveir = 0.0
                    sumirrms = 0.0
                    sumredrms = 0.0
                    SpO2 = 0.0
                    ESpO2 = 90.0                # Initial value
                    i = 0
                    num = 0                    # Calculate once every 30 samples
                    Spo2_ave = 0.0

            ###################################

            # Add the IR reading to the heart rate monitor
            # Note: based on the skin color, the red, IR or green LED can be used
            # to calculate the heart rate with more accuracy.
            hr_monitor.add_sample(ir_reading)

        # Periodically calculate the heart rate every `hr_compute_interval` seconds
        if ticks_diff(ticks_ms(), ref_time) / 1000 > hr_compute_interval:
            # Calculate the heart rate
            heart_rate = hr_monitor.calculate_heart_rate()
            if heart_rate == None:
                print("Not enough data to calculate heart rate")
            #else :
            #    print("Heart Rate: {:.0f} BPM".format(heart_rate))
            # Reset the reference time
            ref_time = ticks_ms()


if __name__ == "__main__":
    main()


