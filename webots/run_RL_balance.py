import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cython')))

import time
import numpy as np
import onnxruntime as ort
from pykondo import Kondo

session = ort.InferenceSession("kondo_balance.onnx")

kondo = Kondo()
if kondo.init() < 0:
    print("Failed to initialize RCB-4 connection.")
    exit()

print("Brain online. Taking over hardware control (Optimized Round-Robin).")

servo_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # Adjust to your 12 leg servos

# Cached sensor values
gyro_pitch = 286
gyro_roll  = 286
accel_z    = 306
accel_y    = 341

poll_counter = 0

while True:
    start_time = time.perf_counter()

    # --- A. Round-Robin Sensor Polling (1 sensor per frame) ---
    try:
        current_sensor = poll_counter % 4
        if current_sensor == 0:
            res = kondo.read_analog(1)
            if res and res[0] >= 0: gyro_pitch = res[1]
        elif current_sensor == 1:
            res = kondo.read_analog(2)
            if res and res[0] >= 0: gyro_roll = res[1]
        elif current_sensor == 2:
            res = kondo.read_analog(3)
            if res and res[0] >= 0: accel_z = res[1]
        elif current_sensor == 3:
            res = kondo.read_analog(4)
            if res and res[0] >= 0: accel_y = res[1]
        poll_counter += 1
    except Exception:
        pass

    # Convert to rad/s and m/s^2 using custom zero-points and multipliers
    gyro_pitch_rads = (gyro_pitch - 286) * 0.127  
    gyro_roll_rads  = (gyro_roll - 286) * 0.127
    accel_z_ms2 = (accel_z - 306) * 0.0493
    accel_y_ms2 = (accel_y - 341) * 0.0484

    # Pack the array for the ONNX Brain
    imu_state = np.array([[accel_z_ms2, accel_y_ms2, 9.81, gyro_roll_rads, gyro_pitch_rads, 0.0]], dtype=np.float32)

    # --- B. Think (ONNX) ---
    actions = session.run(["joint_actions"], {"imu_state": imu_state})[0][0]

    # --- C. Write to Muscles (pykondo) ---
    target_angles = []
    for i in range(len(actions)):
        degrees = actions[i] * 57.2958
        target_angles.append(degrees)
    
    kondo.set_angles(servo_ids[:len(actions)], target_angles, 1.0) 

    # --- D. Maintain exactly 16ms per frame ---
    elapsed = time.perf_counter() - start_time
    if elapsed < 0.016:
        time.sleep(0.016 - elapsed)