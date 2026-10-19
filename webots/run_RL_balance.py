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

print("Brain online. Commencing 14-joint balance control...")

# 14-joint servo mapping: 12 leg joints + 2 shoulders
servo_ids = [
    5, 6, 7, 8, 9, 10,  # Right Leg
    5, 6, 7, 8, 9, 10,  # Left Leg
    1, 1                # Right & Left Shoulders
]

# Cached sensor values
gyro_pitch = 286
gyro_roll  = 286
accel_z    = 306
accel_y    = 341

poll_counter = 0
frame_count = 0
ramp_duration_frames = 60 

previous_filtered_actions = None
SMOOTHING_ALPHA = 0.25 

while True:
    start_time = time.perf_counter()

    # --- A. Round-Robin Sensor Polling ---
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

    # Physical sensor conversions
    gyro_pitch_rads = (gyro_pitch - 286) * 0.127  
    gyro_roll_rads  = (gyro_roll - 286) * 0.127
    accel_z_ms2 = (accel_z - 306) * 0.0493
    accel_y_ms2 = (accel_y - 341) * 0.0484

    # Pack observation (6 floats)
    imu_state = np.array([[accel_z_ms2, accel_y_ms2, 9.81, gyro_roll_rads, gyro_pitch_rads, 0.0]], dtype=np.float32)

    # --- B. Inference ---
    raw_actions = session.run(["joint_actions"], {"imu_state": imu_state})[0][0]

    # --- C. Action Filtering ---
    if previous_filtered_actions is None:
        previous_filtered_actions = np.copy(raw_actions)

    filtered_actions = (SMOOTHING_ALPHA * raw_actions) + ((1.0 - SMOOTHING_ALPHA) * previous_filtered_actions)
    previous_filtered_actions = np.copy(filtered_actions)

    # Soft-start ramp factor
    if frame_count < ramp_duration_frames:
        ramp_factor = float(frame_count) / float(ramp_duration_frames)
        frame_count += 1
    else:
        ramp_factor = 1.0

    # --- D. Sagittal Reflex Bias ---
    ACCEL_THRESHOLD = 1.5
    reflex_bias = [0.0] * len(filtered_actions)

    if abs(accel_y_ms2) > ACCEL_THRESHOLD:
        direction = 1.0 if accel_y_ms2 > 0 else -1.0
        intensity = min(1.0, (abs(accel_y_ms2) - ACCEL_THRESHOLD) / 8.0)
        
        angle_15_rad = 0.26 * direction * intensity
        angle_5_rad  = 0.087 * direction * intensity

        # ID 7: Hips (pitch)
        reflex_bias[2] = angle_15_rad
        reflex_bias[8] = angle_15_rad

        # ID 9: Ankles (pitch)
        reflex_bias[4] = angle_5_rad
        reflex_bias[10] = angle_5_rad

        # ID 1: Shoulders
        reflex_bias[12] = angle_15_rad
        reflex_bias[13] = angle_15_rad

    # --- E. Output to Servos ---
    target_angles = []
    for i in range(len(filtered_actions)):
        total_action = (filtered_actions[i] + reflex_bias[i]) * ramp_factor
        degrees = total_action * 57.2958
        target_angles.append(degrees)

    kondo.set_angles(servo_ids[:len(filtered_actions)], target_angles, 1.0) 

    # --- F. Loop Rate Maintenance (16ms) ---
    elapsed = time.perf_counter() - start_time
    if elapsed < 0.016:
        time.sleep(0.016 - elapsed)