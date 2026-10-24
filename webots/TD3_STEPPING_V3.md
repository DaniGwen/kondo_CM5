# TD3 recovery-step experiment

SAC was gracefully stopped at 1,839,866 steps, preserving model and replay. Repeated evaluation (five identical repetitions per direction) on the box slab, friction 1.0 and leg torque scale 1.25:

| Policy | 4 N / 20 | 5 N / 20 | 6 N / 20 | 8 N / 20 |
|---|---:|---:|---:|---:|
| Neutral | 20 | 10 | 10 | 0 |
| Best SAC v2 (50000) | 20 | 10 | 10 | 0 |
| Latest SAC v2 | 20 | 5 | 5 | 0 |

Best SAC improves one 6 N direction versus the prior learned policy but does not exceed neutral recovery counts, and further training did not improve its early best. This is the practical basis for switching algorithms, not a statistical significance claim. Detailed trials: `runs/sac_v2_stop_evaluation_20261024_01`.

## TD3 continuation

Run `runs/td3_recovery_step_v3_20261024_01`, 2,000,000 additional training steps. Source weights: `runs/adaptive_mixed160_20261021_01/best_model.zip`. Actor, twin critics, and target networks retain prior weights; added observation connections initialize to zero. Validation confirms equal normalized actor and critic outputs before learning. Physical commands can differ because action ranges changed. Optimizers and replay are fresh; legacy six-feature replay is incompatible.

World: static box slab, original surface height/walls, friction 1.0, +25% simulated leg torque, unchanged speed/PID. Observation version 3 is opt-in; versions 1 and 2 remain supported. Do not use this model in the old hardware runner.

Observations 0–50 retain v2 semantics (positions/errors normalize by the new action limits). Additional indices:

| Indices | Features |
|---|---|
| 51–56 | Right then left foot endpoint position relative to root, expressed in current body axes |
| 57–58 | Right/left foot endpoint height change since last push |
| 59–60 | Floor contact for right/left foot or ankle, from root contact ownership and point height |
| 61–62 | Applied push components in body axes at push onset, retained for two-second recovery window |
| 63 | Seconds since push onset |
| 64–65 | Right/left foot world displacement projected onto body-forward axis recorded at push onset |

66 policy observations plus tilt and root height = 68 doubles in each state packet. Push feedback is privileged simulator information, unavailable from the existing hardware sensor path. No future push information is exposed.

## Recovery-step objective

V2 posture/load reward remains. For pushes strictly above 4 N and within 45 degrees of the body longitudinal axis, a foot can earn 0.75 once per event by lifting at least 5 mm, losing contact, translating at least 10 mm along the force direction, then landing within 3 mm of its initial endpoint height with floor contact and body tilt below 0.35 rad. Reward is blocked on falls. Forward force promotes forward stepping; backward force promotes backward stepping. Side pushes do not receive this directional step bonus. The feature is a training incentive, not a scripted gait or guarantee of learned stepping. Sliding without lift/contact loss does not qualify.

Action limits per leg, radians: FemurHead1 0.10, FemurHead2 0.15, Femur 0.25, Crus 0.40, Ankle 0.20, Foot 0.15. Shoulders remain 0.20. Smoothing and per-step delta limits remain. These limits are experimental; hardware feasibility is not established.

Curriculum starts at stage 1, maximum 5 N, retaining the easy/hard mixture and 160 ms pushes. Stronger forces are necessary to activate the above-4-N objective. Independent evaluations/checkpoints every 25000 steps include 4/5/6/8 N directions, rewarded landing counts, and a ten-second no-push hold. Existing recovery/preservation and 60% hip-load gates remain. Exploration noise 0.1 in normalized action space, learning rate 0.0001, warm-up 5000 steps.

Validation: directional landing, threshold, no repeated reward, no reward for sliding/wrong direction; legacy controller tests; v2 protocol and load gates; real v3 neutral hold (hip usage 9.59%), both feet correctly report contact, positive/negative body-force signs checked with scheduled 5 N pushes; expanded TD3 actor/critics match source outputs and complete 192 updates in a 256-step smoke test. No learning result is claimed yet.

Place `STOP` in the run directory for normal model/replay saving. Current launcher creates fresh runs; do not rerun it against an existing directory. Training state is in `td3_status.json`, evaluation history in `evaluations/history.jsonl`, and TensorBoard includes recovery_step_bonus and actuator-load metrics.

Update 2026-10-24: The old-axis run above is historical and stopped. Fresh TD3 now uses the corrected model with fresh weights/replay. See CORRECTED_MODEL.md for the active configuration.
