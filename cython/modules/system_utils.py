import time
import asyncio
from gpiozero import LED
from modules import distance_sensor
from modules.config import *

# Global LED object
_status_led = None

def ensure_pigpiod(start_timeout=2.0):
    """Deprecated on CM5 architecture. Left intact so legacy scripts do not crash."""
    return True

def setup_gpio():
    """Initializes standard Raspberry Pi GPIO pins using modern gpiozero."""
    global _status_led
    
    if _status_led is None:
        # Setup LED and set it HIGH (True) immediately
        _status_led = LED(LED_PIN, initial_value=True)
    
    # Initialize Distance Sensor from module
    distance_sensor.init_sensor(TRIG_PIN, ECHO_PIN)

def cleanup_gpio(ms=None):
    """Safely shuts down servos and cleans up GPIO pins."""
    global _status_led
    if ms:
        ms.stop()
    if _status_led:
        _status_led.close()
        _status_led = None

async def blink_led(duration=3):
    """Asynchronously blinks the LED for a given duration."""
    global _status_led
    if not _status_led:
        return
        
    end_time = time.monotonic() + duration
    while time.monotonic() < end_time:
        _status_led.off()
        await asyncio.sleep(0.1)
        _status_led.on()
        await asyncio.sleep(0.1)
    _status_led.on()