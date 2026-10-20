"""Run an exported TD3 ONNX policy on the physical Kondo KHR-3HV.

SAFETY: hardware movement is DISABLED by default. Before setting
HARDWARE_CONTROL_ENABLED=True, verify the actual 14 unique RCB-4 channels,
calibrated standing-pose commands, servo directions, sensor calibration, and
joint limits. First run in dry-run mode and compare sensor features with Webots.

The matching export_kondo_onnx.py exports outputs in the environment action units:
12 leg offsets limited to +/-0.10 rad and two shoulder offsets limited to +/-0.20 rad.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Keep the repository layout used by the original script (script one folder
# below the Kondo-KHR-3HV project root). Adjust if you place this elsewhere.
PROJECT_CYTHON_PATH = Path(__file__).resolve().parent.parent / "cython"
EXPLICIT_CYTHON_PATH = os.getenv("KONDO_CYTHON_PATH")
if EXPLICIT_CYTHON_PATH:
    sys.path.append(EXPLICIT_CYTHON_PATH)
elif PROJECT_CYTHON_PATH.exists():
    sys.path.append(str(PROJECT_CYTHON_PATH))

from pykondo import Kondo  # noqa: E402

MODEL_PATH = os.getenv("KONDO_ONNX_MODEL", "kondo_balance_v2.onnx")
CONTROL_PERIOD_SECONDS = 0.016
HARDWARE_CONTROL_ENABLED = False  # Explicitly change only after hardware validation.

ACTION_COUNT = 14
ACTION_LIMITS_RAD = np.array([0.10] * 12 + [0.20, 0.20], dtype=np.float32)
ACTION_DELTA_LIMITS_RAD = np.array([0.03] * 12 + [0.04, 0.04], dtype=np.float32)
SMOOTHING_ALPHA = 0.25
RAMP_DURATION_FRAMES = 100
SERVO_COMMAND_SPEED = 1.0  # Preserve the existing pykondo call convention; verify on hardware.

# Fill these only after checking your actual RCB-4 channel map and calibrated stand.
# Use 14 distinct channel IDs, in the same order as the policy joint list below.
SERVO_IDS = []
# Values must be absolute commands in degrees that create your calibrated stable stand.
# Do NOT assume all-zero commands are a valid physical standing pose.
STANDING_ANGLES_DEG = []

# Physical analog-channel calibration retained from your original script.
# These baselines and scales must be measured/confirmed for your robot.
GYRO_PITCH_CHANNEL = 1
GYRO_ROLL_CHANNEL = 2
ACCEL_FORWARD_CHANNEL = 3
ACCEL_LATERAL_CHANNEL = 4
GYRO_BASELINE = 286.0
ACCEL_FORWARD_BASELINE = 306.0
ACCEL_LATERAL_BASELINE = 341.0
GYRO_SCALE_RAD_S_PER_COUNT = 0.127
ACCEL_FORWARD_SCALE_M_S2_PER_COUNT = 0.0493
ACCEL_LATERAL_SCALE_M_S2_PER_COUNT = 0.0484

JOINT_ORDER = [
    "RightFemurHead1", "RightFemurHead2", "RightFemur", "RightCrus", "RightAnkle", "RightFoot",
    "LeftFemurHead1", "LeftFemurHead2", "LeftFemur", "LeftCrus", "LeftAnkle", "LeftFoot",
    "RightShoulder", "LeftShoulder",
]


def read_analog_checked(kondo: Kondo, channel: int, label: str) -> float:
    result = kondo.read_analog(channel)
    if result is None or len(result) < 2 or result[0] < 0:
        raise RuntimeError(f"Failed to read {label} from analog channel {channel}: {result!r}")
    return float(result[1])


def validate_hardware_configuration() -> None:
    if len(SERVO_IDS) != ACTION_COUNT:
        raise RuntimeError("Set SERVO_IDS to the verified 14 physical RCB-4 channels before enabling movement.")
    if len(set(SERVO_IDS)) != ACTION_COUNT:
        raise RuntimeError("SERVO_IDS contains duplicates. Verify the physical channel mapping; independent joints need distinct channels.")
    if len(STANDING_ANGLES_DEG) != ACTION_COUNT:
        raise RuntimeError("Set 14 calibrated STANDING_ANGLES_DEG values before enabling movement.")
    if not np.all(np.isfinite(np.asarray(STANDING_ANGLES_DEG, dtype=float))):
        raise RuntimeError("STANDING_ANGLES_DEG contains non-finite values.")


def main() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path.resolve()}. Run export_kondo_onnx.py first.")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise RuntimeError(f"Expected one model input and at least one output; got {len(inputs)} inputs, {len(outputs)} outputs.")
    input_name = inputs[0].name
    output_name = outputs[0].name
    expected_shape = inputs[0].shape
    if expected_shape and isinstance(expected_shape[-1], int) and expected_shape[-1] != 6:
        raise RuntimeError(f"Expected a 6-feature ONNX input, found shape {expected_shape}.")

    if HARDWARE_CONTROL_ENABLED:
        validate_hardware_configuration()
    else:
        print("DRY RUN: policy inference is enabled, but no servo commands will be sent.", flush=True)

    kondo = Kondo()
    if kondo.init() < 0:
        raise RuntimeError("Failed to initialize RCB-4 connection.")

    print(f"Policy loaded: {model_path}; input={input_name!r}; output={output_name!r}", flush=True)
    print("Observation order: [forward_accel, lateral_accel, 9.81, roll_rate, pitch_rate, 0.0]", flush=True)
    if HARDWARE_CONTROL_ENABLED:
        print("HARDWARE CONTROL ENABLED. Keep physical support and emergency stop available.", flush=True)

    previous_applied = np.zeros(ACTION_COUNT, dtype=np.float32)
    frame_count = 0
    loop_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            # Read all channels once in each control iteration rather than mixing
            # four differently aged round-robin readings into one observation.
            gyro_pitch_raw = read_analog_checked(kondo, GYRO_PITCH_CHANNEL, "gyro pitch")
            gyro_roll_raw = read_analog_checked(kondo, GYRO_ROLL_CHANNEL, "gyro roll")
            accel_forward_raw = read_analog_checked(kondo, ACCEL_FORWARD_CHANNEL, "forward acceleration")
            accel_lateral_raw = read_analog_checked(kondo, ACCEL_LATERAL_CHANNEL, "lateral acceleration")

            gyro_pitch = (gyro_pitch_raw - GYRO_BASELINE) * GYRO_SCALE_RAD_S_PER_COUNT
            gyro_roll = (gyro_roll_raw - GYRO_BASELINE) * GYRO_SCALE_RAD_S_PER_COUNT
            accel_forward = (accel_forward_raw - ACCEL_FORWARD_BASELINE) * ACCEL_FORWARD_SCALE_M_S2_PER_COUNT
            accel_lateral = (accel_lateral_raw - ACCEL_LATERAL_BASELINE) * ACCEL_LATERAL_SCALE_M_S2_PER_COUNT

            # Must match kondo_tcp_server.py's six training features, including
            # constant placeholders for vertical acceleration and yaw rate.
            obs = np.asarray(
                [accel_forward, accel_lateral, 9.81, gyro_roll, gyro_pitch, 0.0],
                dtype=np.float32,
            )
            model_actions = session.run([output_name], {input_name: obs.reshape(1, 6)})[0]
            raw_actions = np.asarray(model_actions, dtype=np.float32).reshape(-1)
            if raw_actions.size != ACTION_COUNT or not np.all(np.isfinite(raw_actions)):
                raise RuntimeError(f"Invalid ONNX action output; expected {ACTION_COUNT} finite values, got shape {model_actions.shape}.")

            # The supplied export wrapper returns environment action units (radians).
            raw_actions = np.clip(raw_actions, -ACTION_LIMITS_RAD, ACTION_LIMITS_RAD)
            if frame_count < RAMP_DURATION_FRAMES:
                ramp_factor = float(frame_count + 1) / float(RAMP_DURATION_FRAMES)
                frame_count += 1
                raw_actions = raw_actions * ramp_factor

            smoothed = SMOOTHING_ALPHA * raw_actions + (1.0 - SMOOTHING_ALPHA) * previous_applied
            delta = np.clip(smoothed - previous_applied, -ACTION_DELTA_LIMITS_RAD, ACTION_DELTA_LIMITS_RAD)
            applied_actions = (previous_applied + delta).astype(np.float32)
            previous_applied = applied_actions.copy()

            if HARDWARE_CONTROL_ENABLED:
                target_angles_deg = (
                    np.asarray(STANDING_ANGLES_DEG, dtype=np.float32)
                    + np.rad2deg(applied_actions)
                ).tolist()
                kondo.set_angles(SERVO_IDS, target_angles_deg, SERVO_COMMAND_SPEED)

            loop_count += 1
            if loop_count % 30 == 0:
                elapsed = time.perf_counter() - loop_start
                print(
                    f"loop={loop_count} elapsed={elapsed * 1000.0:.1f} ms "
                    f"features={np.round(obs, 3).tolist()} "
                    f"max_offset={np.max(np.abs(applied_actions)):.4f} rad "
                    f"mode={'CONTROL' if HARDWARE_CONTROL_ENABLED else 'DRY-RUN'}",
                    flush=True,
                )

            elapsed = time.perf_counter() - loop_start
            if elapsed < CONTROL_PERIOD_SECONDS:
                time.sleep(CONTROL_PERIOD_SECONDS - elapsed)
            elif loop_count % 30 == 0:
                print("Warning: sensor reads/inference exceeded the 16 ms target; actual loop is slower.", flush=True)

    except KeyboardInterrupt:
        print("Stopped by user. Check the robot's physical state before approaching.", flush=True)
    except Exception as exc:
        print(f"Balance loop stopped due to error: {exc}", flush=True)
        if HARDWARE_CONTROL_ENABLED:
            print("No further policy commands will be sent. Use the robot's normal safe stop procedure.", flush=True)
        raise
    finally:
        print(f"Elapsed wall time: {time.perf_counter() - start_time:.1f} s", flush=True)


if __name__ == "__main__":
    main()
