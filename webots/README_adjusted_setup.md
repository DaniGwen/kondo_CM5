# Kondo KHR-3HV TD3 balance — adjusted scripts

This bundle updates the Webots server, Gymnasium training environment, ONNX export, and physical-robot inference runner as a matched set.

## Files

- `kondo_tcp_server.py` — Webots Supervisor controller. Uses a request/response TCP protocol and returns six policy observations plus two simulation-only telemetry values (`tilt_rad`, `height_m`).
- `train_kondo_balance.py` — TD3 Gymnasium environment with an upright-posture reward, action filtering/rate limiting, episode metrics, and a new model/replay-buffer name.
- `export_kondo_onnx.py` — exports the trained Stable-Baselines3 TD3 actor with the expected ONNX input/output names and action scaling.
- `run_RL_balance.py` — physical Kondo inference loop. **Motor commands are disabled by default.** It uses matching six-feature observations and filtering, but requires verified hardware calibration before control is enabled.

## Important assumptions to verify

1. **Webots coordinates:** this version assumes a Z-up world, a robot root whose local Z axis is vertical at the standing pose, and a nominal root height of 0.24 m. Check the actual `.wbt`/PROTO model. Change `RESET_POSITION`, `read_telemetry()`, and the fall-height threshold if your model differs.
2. **Observation axes and signs:** the six inputs are `[forward_accel, lateral_accel, 9.81, roll_rate, pitch_rate, 0.0]`. In Webots, the first two features currently use `accelerometer[0:2]` and the next two rates use `gyro[0:2]`. The physical script maps these features to analog channels 3, 4, 2, and 1 using the scales/baselines retained from your previous script. **These mappings are not verified by the uploaded code.** Check the sensor datasheets/RCB-4 wiring and confirm axes, signs, units, and neutral baselines before transferring a policy.
3. **Joint order:** both simulation and policy expect the 14 joints in the order listed in `JOINT_ORDER` in the runner and `motor_names` in the server. Ensure each output corresponds to the intended physical joint and sign.
4. **Standing posture:** Webots uses `READY_STANCE_RAD = [0.0] * 14`, matching your previous script. Change it if zero is not a feasible neutral stand in your Webots model. On the physical robot, fill `STANDING_ANGLES_DEG` with 14 calibrated absolute commands that reproduce a stable stand using your `pykondo` API.
5. **Servo channels:** the original `servo_ids` array repeats IDs for both legs and shoulders. The adjusted runner intentionally refuses hardware control unless you configure 14 distinct channel IDs. I cannot infer the correct mapping safely from the supplied files; verify the actual RCB-4 channel wiring and joint mapping first.
6. **Reward/fall thresholds:** the initial values `FALL_TILT_RAD=0.85`, `FALL_HEIGHT_M=0.13`, and push force curriculum are starting points. Inspect recorded Webots tilt/height and adapt the thresholds to the actual model before trusting success/fall statistics.

## Train in Webots

1. Back up your Webots world/controller files.
2. Replace the Webots controller with the adjusted `kondo_tcp_server.py` (or copy its contents into the controller file assigned to the robot). Keep the motor device names aligned with the Webots model.
3. Start the world. The controller should report that it is listening on port 5555.
4. In the Python environment where `gymnasium`, `stable-baselines3`, `numpy`, and `torch` are installed, run:

   ```bash
   python train_kondo_balance.py
   ```

   If Webots runs on another machine, set `KONDO_WEBOTS_HOST` to that machine's IP. Open port 5555 only on a trusted local network.
5. Use TensorBoard if installed:

   ```bash
   tensorboard --logdir ./kondo_tensorboard_v2/
   ```

6. When training finishes or is interrupted normally with Ctrl+C, the script attempts to save `kondo_td3_v2.zip` and `kondo_td3_v2_replay_buffer.pkl`. Resume only with this adjusted v2 environment. The old `kondo_td3_smooth.zip` is deliberately not loaded because its replay transitions used the previous protocol/reward.

The controller uses one client connection. Do not start multiple training processes against the same Webots server. Training/evaluation should be performed in simulation before any real robot test.

## Export the learned policy

In the environment with the same Stable-Baselines3/PyTorch versions, run:

```bash
python export_kondo_onnx.py --model kondo_td3_v2.zip --output kondo_balance_v2.onnx
```

The exported model contract is:

- input name: `imu_state`, shape `[batch, 6]`, float32
- output name: `joint_actions`, shape `[batch, 14]`, float32
- outputs are already scaled to the training environment's action units: radians, with 12 leg actions limited to ±0.10 rad and two shoulder actions limited to ±0.20 rad

Before any physical movement, compare `model.predict(obs, deterministic=True)` with ONNX outputs for a small set of recorded observations. They should match within normal floating-point tolerance.

## Physical runner — dry-run first

1. Place `run_RL_balance.py` where it can import your existing `pykondo` package. The default repository lookup assumes this script is one directory below the project root. Or set `KONDO_CYTHON_PATH` to the directory containing the built `pykondo` module.
2. Place `kondo_balance_v2.onnx` where the script can find it, or set `KONDO_ONNX_MODEL` to its full path.
3. Leave `HARDWARE_CONTROL_ENABLED = False`. Run the script and check that readings are fresh, the loop timing is acceptable, and all six policy features have the expected signs and order.
4. Verify the neutral baseline/scales with a stationary robot and confirm all sensors return valid readings. The current conversion constants are copied from the old runner; they have not been independently verified.
5. Only after verifying the robot's physical channel map and the correct calibrated standing commands, enter the 14 unique `SERVO_IDS`, 14 `STANDING_ANGLES_DEG`, inspect all joint limits, and consider enabling hardware control under direct supervision with the robot physically supported and a reachable stop control.

The hardware runner stops issuing new commands if sensor reads or inference fail. It does not guess at an RCB-4 torque-disable API, and it cannot guarantee that servos will release torque on process exit. Use the robot's normal safe-stop procedure when testing.

## What changed from the original files

- Full TCP packet reads/writes (`sendall`/exact receive) and one action → one stepped observation.
- Reset is an explicit request/response, avoiding stale state packets after reset.
- The simulation returns tilt and root-height telemetry for reward/termination, but only the six deployable features are fed to the actor.
- Removed the hand-coded sagittal reflex so simulation and physical inference do not apply different hidden corrections.
- Reward directly favors upright posture and survival and mildly penalizes excessive action/rate; shoulder penalty now targets shoulders only by design because the arbitrary shoulder/leg movement rewards were removed.
- Reduced the initial batch size to 256, added a warm-up phase, metrics, deterministic disturbance seed, and staged push-strength curriculum.
- A separate ONNX export script guarantees the 14 outputs are scaled into the environment action units.
- The physical runner is dry-run by default and will not move the robot until explicitly configured and enabled.

These scripts are syntax-checked, but have **not** been run against your particular Webots world or physical KHR-3HV hardware. Exact sensor-axis calibration, root-height assumptions, actuator mapping, and standing-pose commands still need verification on your setup.
