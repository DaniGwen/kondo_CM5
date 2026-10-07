import RPi.GPIO as GPIO
import time
import sys

# --- Pin Definitions (BCM Naming) ---
# Sensor Trigger Pin (Output)
TRIG_PIN = 24 # Physical Pin 18
# Sensor Echo Pin (Input)
ECHO_PIN = 23 # Physical Pin 16

# LED Pin (Output)
# NOTE: MUST be connected to the LED via a CURRENT-LIMITING RESISTOR (e.g., 220 Ohm).
LED_PIN = 17 # Physical Pin 11

# --- Constants ---
SPEED_OF_SOUND = 34300 # cm/s (speed of sound at sea level)
DISTANCE_THRESHOLD = 15.0 # cm

def setup():
    """Initializes GPIO settings."""
    print("Setting up GPIO pins...")
    GPIO.setmode(GPIO.BCM)
    
    # Setup sensor pins
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    
    # Setup LED pin
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW) # Start with LED off
    
    # Ensure trigger pin is low before starting
    GPIO.output(TRIG_PIN, GPIO.LOW)
    time.sleep(0.5)

def measure_distance():
    """
    Measures distance using the HC-SR04 sensor.
    """
    # 1. Send the Trigger pulse
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.000010) # 10 microsecond pulse
    GPIO.output(TRIG_PIN, GPIO.LOW)
    
    pulse_start = time.time()
    pulse_end = time.time()
    
    # 2. Wait for the Echo signal (Rising edge)
    # Timeout check is crucial for stability
    start_time = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > 0.05: # Max wait 50ms (for ~8.5m range)
            return 999.0 # Timeout

    # 3. Wait for the pulse to end (Falling edge)
    start_time = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if pulse_end - start_time > 0.05: # Max wait 50ms
            return 999.0 # Timeout
        
    # Calculate duration and distance
    pulse_duration = pulse_end - pulse_start
    if pulse_duration <= 0:
        return 999.0 # Invalid duration

    # Distance = (time in seconds * speed of sound) / 2 (out and back)
    distance = (pulse_duration * SPEED_OF_SOUND) / 2
    
    return distance

def loop():
    """Main loop for distance checking and LED control."""
    try:
        print(f"Starting HC-SR04 measurement. LED on BCM {LED_PIN} (Pin 11).")
        print(f"Trigger: BCM {TRIG_PIN} (Pin 18), Echo: BCM {ECHO_PIN} (Pin 16).")
        print(f"Threshold: {DISTANCE_THRESHOLD} cm.")
        
        while True:
            dist = measure_distance()
            
            # --- LED Control Logic ---
            if 2.0 < dist < DISTANCE_THRESHOLD:
                # Valid distance within range
                status = "OBJECT DETECTED"
                GPIO.output(LED_PIN, GPIO.HIGH)
            elif dist == 999.0:
                # Measurement failed (timeout)
                status = "Measurement Error/Timeout"
                GPIO.output(LED_PIN, GPIO.LOW)
            else:
                # Clear or out of sensor range (> 400 cm)
                status = "Clear"
                GPIO.output(LED_PIN, GPIO.LOW)
            
            # Print results (with a safety filter for printing extreme junk values)
            print(f"Distance: {dist:.1f} cm | Status: {status}")
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        GPIO.cleanup() # Clean up all GPIO settings on exit

if __name__ == "__main__":
    try:
        setup()
        loop()
    except Exception as e:
        print(f"Setup failed: {e}")
        GPIO.cleanup()