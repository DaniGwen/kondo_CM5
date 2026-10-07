import time
import asyncio
from gpiozero import DistanceSensor

# Global sensor object
_sensor = None

def init_sensor(trig_pin, echo_pin):
    """Initializes the ultrasonic sensor using modern gpiozero."""
    global _sensor
    if _sensor is None:
        _sensor = DistanceSensor(echo=echo_pin, trigger=trig_pin, max_distance=4.0)

def _read_distance_sync():
    """Synchronous function to safely read the sensor without blocking the async loop."""
    global _sensor
    if _sensor:
        try:
            return _sensor.distance * 100.0
        except Exception:
            return 999.0
    return 999.0

async def distance_task(state, trig_pin, echo_pin):
    """Background task to continuously poll the sensor and update the shared state."""
    loop = asyncio.get_running_loop()
    print("Starting distance sensor (gpiozero)...")
    last_print = time.time()
    
    last_valid_dist = 999.0 
    
    while True:
        # Safely execute the blocking read in a separate background thread
        dist = await loop.run_in_executor(None, _read_distance_sync)
        
        if dist >= 390.0:
            dist = 999.0

        if dist >= 999.0 and last_valid_dist < 999.0:
            dist = last_valid_dist
        else:
            last_valid_dist = dist

        state["distance_cm"] = dist
        
        if time.time() - last_print > 1.0:
            print(f"[DEBUG] Distance Sensor: {dist:.1f} cm")
            last_print = time.time()
            
        await asyncio.sleep(0.05)