import time
import asyncio
from gpiozero import DistanceSensor

# Global sensor object
_sensor = None

def init_sensor(trig_pin, echo_pin):
    """Initializes the ultrasonic sensor using modern gpiozero."""
    global _sensor
    if _sensor is None:
        # max_distance=4.0 meters limits the timeout ping. 
        _sensor = DistanceSensor(echo=echo_pin, trigger=trig_pin, max_distance=4.0)

async def distance_task(state, trig_pin, echo_pin):
    """Background task to continuously poll the sensor and update the shared state."""
    global _sensor
    print("Starting distance sensor (gpiozero)...")
    last_print = time.time()
    
    # Memory to filter out physical blind-spot glitches (999.0)
    last_valid_dist = 999.0 
    
    while True:
        if _sensor:
            # gpiozero returns distance in meters, convert to cm
            raw_m = _sensor.distance
            dist = raw_m * 100.0
            
            # If it reads near max distance (390+ cm), treat as out-of-range (999.0)
            if dist >= 390.0:
                dist = 999.0

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
            
        # We no longer need run_in_executor because gpiozero updates the distance property asynchronously
        await asyncio.sleep(0.05)