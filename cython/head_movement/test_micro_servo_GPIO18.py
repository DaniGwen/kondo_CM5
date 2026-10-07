#!/usr/bin/env python3
# servo_small_sweep.py
# Requires: sudo apt install pigpio python3-pigpio
# Start pigpio daemon before running: sudo systemctl start pigpiod

import time
import pigpio
import signal
import sys

GPIO = 18               # BCM pin 18 (physical pin 12)
CENTER_US = 1200       # center pulse width in microseconds
DELTA_US = 700          # small angle offset (+/-)
STEP_US = 50            # smooth step in microseconds
DELAY = 0.02            # delay between steps (seconds)
CYCLES = 6              # how many up/down cycles to run

pi = pigpio.pi()
if not pi.connected:
    print("ERROR: pigpio daemon not running. Start with: sudo systemctl start pigpiod")
    sys.exit(1)

def cleanup(signum=None, frame=None):
    try:
        pi.set_servo_pulsewidth(GPIO, 0)  # stop pulses
    finally:
        pi.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    # center first
    pi.set_servo_pulsewidth(GPIO, CENTER_US)
    time.sleep(0.5)

    for _ in range(CYCLES):
        # move from center to center+DELTA smoothly
        target = CENTER_US + DELTA_US
        for us in range(CENTER_US, target + 1, STEP_US):
            pi.set_servo_pulsewidth(GPIO, us)
            time.sleep(DELAY)
        time.sleep(0.12)

        # move back to center
        for us in range(target, CENTER_US - 1, -STEP_US):
            pi.set_servo_pulsewidth(GPIO, us)
            time.sleep(DELAY)
        time.sleep(0.12)

        # move to center-DELTA smoothly
        target = CENTER_US - DELTA_US
        for us in range(CENTER_US, target - 1, -STEP_US):
            pi.set_servo_pulsewidth(GPIO, us)
            time.sleep(DELAY)
        time.sleep(0.12)

        # back to center
        for us in range(target, CENTER_US + 1, STEP_US):
            pi.set_servo_pulsewidth(GPIO, us)
            time.sleep(DELAY)
        time.sleep(0.12)

    # finish centered
    pi.set_servo_pulsewidth(GPIO, CENTER_US)
    time.sleep(0.3)

finally:
    cleanup()
