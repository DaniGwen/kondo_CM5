import time
import asyncio
import RPi.GPIO as GPIO

# Constants
SPEED_OF_SOUND = 34300
MAX_WAIT_TIME_MS = 50

def init_sensor(trig_pin, echo_pin):
    """Initializes the GPIO pins for the ultrasonic sensor."""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(trig_pin, GPIO.OUT)
    GPIO.setup(echo_pin, GPIO.IN)
    GPIO.output(trig_pin, GPIO.LOW)

def measure_distance(trig_pin, echo_pin):
    """Fires the ultrasonic pulse and measures the return time."""
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.000010)
    GPIO.output(trig_pin, GPIO.LOW)
    
    # Wait for the pulse to start
    if GPIO.wait_for_edge(echo_pin, GPIO.RISING, timeout=MAX_WAIT_TIME_MS) is None:
        return 999.0
    pulse_start = time.monotonic()
    
    # Wait for the pulse to return
    if GPIO.wait_for_edge(echo_pin, GPIO.FALLING, timeout=MAX_WAIT_TIME_MS) is None:
        return 999.0
    pulse_end = time.monotonic()
    
    return ((pulse_end - pulse_start) * SPEED_OF_SOUND) / 2

async def distance_task(state, trig_pin, echo_pin):
    """Background task to continuously poll the sensor and update the shared state."""
    loop = asyncio.get_running_loop()
    print("Starting distance sensor...")
    last_print = time.time()
    
    # Memory to filter out physical blind-spot glitches (999.0)
    last_valid_dist = 999.0 
    
    while True:
        # Run the blocking measurement safely in a background thread
        dist = await loop.run_in_executor(None, measure_distance, trig_pin, echo_pin)
        
        # If the sensor glitches (returns 999), use the last known good distance
        if dist >= 999.0 and last_valid_dist < 999.0:
            dist = last_valid_dist
        else:
            last_valid_dist = dist

        state["distance_cm"] = dist
        
        # Debug output every 1 second
        if time.time() - last_print > 1.0:
            print(f"[DEBUG] Distance Sensor: {dist:.1f} cm")
            last_print = time.time()
            
        await asyncio.sleep(0.01)