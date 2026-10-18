import sys
import os
# Add the cython folder to the Python path so it can find pykondo.so
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

# Ensure these 16 IDs match the exact order your AI expects the joints
servo_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

while True:
    start_time = time.perf_counter()

    # --- A. Read Senses (via RCB-4) ---
    # The [1] extracts the actual analog value from the (status, value) tuple
    raw_gyro_pitch = kondo.read_analog(1)[1] 
    raw_gyro_roll  = kondo.read_analog(2)[1] 
    raw_accel_z    = kondo.read_analog(3)[1] 
    raw_accel_y    = kondo.read_analog(4)[1] 

    # Convert to rad/s and m/s^2 using your custom zero-points and multipliers
    gyro_pitch_rads = (raw_gyro_pitch - 286) * 0.127  
    gyro_roll_rads  = (raw_gyro_roll - 286) * 0.127
    accel_z_ms2 = (raw_accel_z - 306) * 0.0493
    accel_y_ms2 = (raw_accel_y - 341) * 0.0484

    # Pack the array for the ONNX Brain
    imu_state = np.array([[accel_z_ms2, accel_y_ms2, 9.81, gyro_roll_rads, gyro_pitch_rads, 0.0]], dtype=np.float32)

    # --- B. Think (ONNX) ---
    # The actions array will return values between -0.1 and 0.2 (radians)
    actions = session.run(["joint_actions"], {"imu_state": imu_state})[0][0]

    # --- C. Write to Muscles (pykondo) ---
    target_angles = []
    for i in range(len(actions)):
        # Convert safely bounded radians directly to degrees
        degrees = actions[i] * 57.2958
        target_angles.append(degrees)
    
    # Send the converted degrees to the active physical servos
    kondo.set_angles(servo_ids[:len(actions)], target_angles, 1.0)

    # --- D. Maintain exactly 16ms per frame ---
    elapsed = time.perf_counter() - start_time
    if elapsed < 0.016:
        time.sleep(0.016 - elapsed)