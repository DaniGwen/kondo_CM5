# Exporting a policy to the real Kondo

## Export is different from robot readiness

The dashboard has a research-package export and a separate gated real-robot export. Neither sends commands, uploads files or enables servo movement. A good simulation score alone is insufficient for physical use. The current best checkpoint may still be the initial baseline; the UI displays its actual saved learning-step count, not the running trainer's step count.

Select a run's best_model.zip. A package contains a deterministic TD3 or SAC ONNX actor, the exact source checkpoint, interface/action bounds, SHA-256 hashes, ONNX numerical parity results, configuration and this guide. ONNX graph validation and parity against model.predict are mandatory. ONNX actions are scaled to radians; they are not raw [-1,1] actor outputs. Filtering and servo conversion happen outside the exported actor. Export never truncates or pads observation inputs to make them appear compatible.

The research button can package a simulator policy for inspection even when hardware export is blocked. Its README and manifest explicitly mark it RESEARCH ONLY. A successful ONNX parity check proves numerical equivalence to the source policy, not balance performance or hardware safety.

## Current observation mismatch

run_RL_balance.py supplies six calibrated IMU features. Current stepping policies require 66 features, including posture, joint position/torque/tracking, foot positions/contact, externally applied force, push age and displacement since the push. The real runner does not supply those. Do not zero-fill the missing features or copy a 66-input ONNX file into the six-input runner.

The preferred deployment experiment is a policy trained using only measurements available on the real robot. Confirm available servo position/current feedback and contact sensors first; do not assume the RCB interface provides all of them. Posture needs an estimator with consistent axes/units/latency. Simulator-applied force and exact contact state are privileged data: remove them from the deployable actor, use validated estimates, or train a hardware-input student policy and re-evaluate it. New sensor/reward definitions require new compatible training data.

The hardware runner also needs fourteen unique verified servo IDs, calibrated standing commands, directions, offsets and mechanical travel limits. Current SERVO_IDS and STANDING_ANGLES_DEG are empty. Hardware control remains disabled. Current step action limits exceed the old runner's ±0.10 rad leg limits, so matching the input count alone is insufficient: validate and implement the same approved action limits, 16 ms control timing, smoothing, delta limits, shoulder filter and radians-to-servo conversion. Do not simply widen real limits without checking mechanics.

## The +25% simulated torque

Our model uses 1.716125 N m per leg motor, compared with the former model value 1.3729 N m. That former value is not a measurement of your installed servos. Exporting changes neither simulated nor real motor strength. Scaling joint-angle commands by 0.8 does not compensate for missing actuator torque.

Before deployment, measure or establish conservative available torque, speed and control behavior for the installed servos at the actual supply voltage, posture, load and temperature. Evaluate the exact candidate at these limits on the corrected model, including every actuator's peak load and saturation duration, foot slip, no-push holds and directional pushes. A policy dependent on the boosted torque needs fine-tuning/retraining at realistic limits and then re-evaluation. Domain-randomized torque/friction/mass/latency can help robustness, but the low-torque cases still need to pass the chosen acceptance criteria. Do not raise hardware voltage/current limits to match the simulator without manufacturer-supported limits.

Suggested sequence:

1. Build a hardware-available observation contract and verify every joint direction/zero/travel independently.
2. Configure measured conservative actuator limits in a separate corrected world and train/evaluate with compatible observations.
3. Repeat the recovery suite over varied poses/timing; require improvement over neutral hold and document required push levels and acceptable failures. Check per-motor reserve, not only the average hip-load gate.
4. Export ONNX and verify parity on representative recorded observations. Measure inference and full-loop timing on the CM5.
5. Replay logged sensor data with hardware output disabled; verify output scale, filtering and limits. Then test a supported robot with constrained motions and an emergency stop, beginning with no pushes.

## Readiness report

The UI's hardware export remains blocked until the checkpoint is trained, belongs to a corrected-model run, matches the existing hardware input contract, has configured servo/standing mapping, and has a checkpoint-specific deployment_validation.json. This is a documented evidence gate, not automatic hardware certification.

The report must bind model_sha256 to the exact best_model.zip and set these fields to true only after the corresponding tests pass:

```
nominal_torque_passed
sensor_contract_passed
servo_mapping_passed
supported_dry_run_passed
recovery_suite_passed
```

For each field, evidence[field] must contain file (a relative path inside the run) and sha256 (the evidence file hash). Evidence should include test configuration, measured limits, trial data, explicit acceptance criteria and results. Copying an approval from another model or editing a checkbox is not validation. If the checkpoint changes, its report no longer matches. The UI still cannot verify the physical truth of a manually authored report; responsible review remains necessary.

At present the runner supports only the six-feature contract. Adding another hardware contract requires actual matching inference code and tests; merely editing this report will not unlock a 66-feature model.

Packages are written into webots/exports/ next to this guide's scripts (the kondo_CM5/webots folder), and can be downloaded from the dashboard. Project-local .export_tools holds ONNX libraries separately from training dependencies. Existing legacy export_kondo_onnx.py remains the old TD3/six-feature exporter; use the dashboard's package exporter for dynamic shapes and deterministic SAC support.

Technical export reference: https://stable-baselines3.readthedocs.io/en/master/guide/export.html
