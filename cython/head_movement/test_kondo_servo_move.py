#!/usr/bin/env python3
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pykondo import Kondo


def main():
    print("Init Kondo...")
    k = Kondo()
    ret = k.init()
    print("init returned:", ret)
    if ret < 0:
        print("Init failed; exit")
        return

    sid = 0
    angle = 20.0
    speed = 0.6
    print(f"Moving servo id {sid} -> angle {angle} speed {speed}")
    r = k.set_angle(sid, angle, speed)
    print("set_angle returned:", r)
    time.sleep(2.0)

    print("Free servo", sid)
    k.free_servo(sid)
    time.sleep(0.2)

    print("Purge and close")
    k.purge()
    k.close()
    print("Done")

if __name__ == '__main__':
    main()
