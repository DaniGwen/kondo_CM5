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

print("Brain online. Taking over hardware control.")

servo_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Cache default resting values in case a serial read times out
last_gyro_pitch = 286
last_gyro_roll  = 286
last_accel_z    = 306
last_accel_y    = 341

while True:
    start_time = time.perf_counter()

    # --- A. Read Senses (via RCB-4 with error protection) ---
    try:
        res1 = kondo.read_analog(1)
        if res1 and res1[0] >= 0: last_gyro_pitch = res1[1]
        
        res2 = kondo.read_analog(2)
        if res2 and res2[0] >= 0: last_gyro_roll = res2[1]
        
        res3 = kondo.read_analog(3)
        if res3 and res3[0] >= 0: last_accel_z = res3[1]
        
        res4 = kondo.read_analog(4)
        if res4 and res4[0] >= 0: last_accel_y = res4[1]
    except Exception:
        # Silently catch serial hiccups to maintain 16ms timing
        pass

    # Convert to rad/s and m/s^2 using custom zero-points and multipliers
    gyro_pitch_rads = (last_gyro_pitch - 286) * 0.127  
    gyro_roll_rads  = (last_gyro_roll - 286) * 0.127
    accel_z_ms2 = (last_accel_z - 306) * 0.0493
    accel_y_ms2 = (last_accel_y - 341) * 0.0484

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