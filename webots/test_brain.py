import onnxruntime as ort
import numpy as np
import time

# 1. Load the model (Make sure the filename matches your GitHub repo)
session = ort.InferenceSession("kondo_balance.onnx")

# 2. Create a dummy IMU state: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
dummy_state = np.zeros((1, 6), dtype=np.float32)

# 3. Warm-up run (ONNX Runtime takes a fraction of a second longer on the very first pass)
session.run(["joint_actions"], {"imu_state": dummy_state})

# 4. Speed Test
start_time = time.perf_counter()
actions = session.run(["joint_actions"], {"imu_state": dummy_state})[0]
end_time = time.perf_counter()

inference_ms = (end_time - start_time) * 1000

print("Calculated 16 Joint Adjustments:")
print(actions[0])
print(f"Inference Speed: {inference_ms:.3f} ms")