# Posture and actuator-load SAC experiment

Run: `runs/sac_posture_load_v2_20261023_01`, target 2,000,000 steps. Fresh SAC weights and replay; box slab, friction 1.0, 25% higher leg torque. Actions, motor speed/PID, action filtering, and fall thresholds retain the preceding experiment's settings.

Set `KONDO_OBSERVATION_VERSION=2` on both simulator and client. The response contains 53 little-endian doubles: 51 observations followed by tilt and root height. Version 1 remains the default six-observation/eight-double protocol for old policies.

Zero-based observation layout:

| Indices | Features |
|---|---|
| 0–5 | Existing acceleration, gravity constant, roll/pitch rates, zero marker |
| 6–8 | World up expressed in body coordinates, from root rotation matrix row 3 |
| 9–22 | Fourteen measured joint positions divided by their action limits |
| 23–36 | Fourteen signed motor torque feedback values divided by available max torque |
| 37–50 | Fourteen target-minus-measured joint errors divided by action limits |

Sensor names `diag_<motor>` are provided by the dedicated v2 world. The six legacy features and scheduled-reset acknowledgement marker retain their ordering. This observation schema uses simulator posture and motor feedback; the existing hardware runner cannot execute these policies without equivalent sensor mapping and a matching inference implementation.

Reward retains survival, exponential upright reward, angular-rate penalty, command-effort penalty, smoothness penalty, and fall penalty. Added per-step costs over twelve leg motors:

- 0.75 times mean squared normalized absolute motor torque, clipped to 2.
- 0.75 times mean squared saturation score: clipped `(absolute normalized torque - 0.8) / 0.2`, range 0–1.
- 0.25 times mean squared normalized tracking error, clipped to ±2.

Evaluations/checkpoints every 25,000 steps. Each evaluation includes the existing directional push suite and a 625-step no-push hold; mean hip usage is scored over the final 125 steps. Promotion and curriculum advancement require no hold fall and mean absolute hip torque fraction at most 0.6, in addition to the existing reference-preservation and recovery gates. The initial model is still saved as a baseline even if it fails this gate; consult `load_gate_passed` before treating `best_model.zip` as a qualified checkpoint.

Training logs include hip and leg torque fractions and tracking error. Evaluation history records no-push hold load and gate status. These coefficients and the 60% gate are experimental choices, not hardware calibration.

Validation: protocol round trip and telemetry offsets, reward ordering/sign symmetry, saturated versus low-load promotion gates, legacy SAC save/resume and controller tests. Real Webots neutral hold: 625 steps, no fall, mean hip fraction 0.0959. Short learning check: 256 steps and 192 gradient updates with finite observations and rewards.

Stop through the run directory's `STOP` file for normal model/replay saving. Resume with `launch_sac_clean_reset.py --run-dir ./runs/sac_posture_load_v2_20261023_01 --resume --observation-version 2 --steps 2000000`. Launcher steps are the total target; observation versions cannot be mixed when resuming.
