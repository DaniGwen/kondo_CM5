import time
import asyncio
import shutil
import subprocess
import RPi.GPIO as GPIO
from modules import distance_sensor
from modules.config import *

def ensure_pigpiod(start_timeout=2.0):
    """Checks if the pigpio daemon is running, and starts it if not."""
    try:
        subprocess.check_call(["pgrep", "pigpiod"], stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        pass

    print("Starting pigpiod...")
    pigpiod_path = shutil.which("pigpiod")
    if pigpiod_path:
        subprocess.Popen(
            ["sudo", pigpiod_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(start_timeout)
        return True
    return False

def setup_gpio():
    """Initializes standard Raspberry Pi GPIO pins."""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    
    # Setup LED
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.HIGH)
    
    # Initialize Distance Sensor from module
    distance_sensor.init_sensor(TRIG_PIN, ECHO_PIN)

def cleanup_gpio(ms=None):
    """Safely shuts down servos and cleans up GPIO pins."""
    if ms:
        ms.stop()
    GPIO.cleanup()

async def blink_led(duration=3):
    """Asynchronously blinks the LED for a given duration."""
    end_time = time.monotonic() + duration
    while time.monotonic() < end_time:
        GPIO.output(LED_PIN, GPIO.LOW)
        await asyncio.sleep(0.1)
        GPIO.output(LED_PIN, GPIO.HIGH)
        await asyncio.sleep(0.1)
    GPIO.output(LED_PIN, GPIO.HIGH)