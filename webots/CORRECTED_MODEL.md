# Corrected training model — 2026-10-24

Active fresh-run world: webots/kondo_RL/worlds/kondoKHR3-HV-box-slab-corrected.wbt.

Changes: LeftFemur, RightFemur, LeftCrus, RightCrus and RightAnkle now rotate about local Y. All 22 saved joint coordinates are neutral, with consistent neutral endpoint transforms. The initial robot root is upright at (0,0,0.24), matching the clean-reset pose. The other joint axes, anchors, motor limits, masses and collision geometry are retained.

Retained settings: static box slab with top z=-0.1; effective friction 1.0; 12 leg motor torque limits 1.716125 N m; ten upper-body torque limits 1.3729 N m; normal default gravity; selfCollision TRUE; kondo_tcp_server controller; fourteen position sensors for observation v2/v3. Version selection remains through KONDO_OBSERVATION_VERSION.

Validation: isolated original/candidate sweep covered twelve leg joints at four nonzero angles; corrected axes passed all 48 checks. Normal-gravity corrected-world check passed 625 neutral-hold steps without falling, both foot contacts true, mean hip torque fraction 0.095922, mean reward 2.996697. Scheduled positive/negative 5 N force signs passed. A fresh TD3 smoke model completed 256 steps and 192 updates. Six geometry, stepping and controller regression tests passed. These are simulator checks, not hardware calibration or evidence of a learned walking gait.

Fresh TD3 launcher td3_stepping_experiment.py now builds/selects this world, creates runs/td3_corrected_step_v3_20261024_01, and starts with fresh weights, optimizer and replay. The above-4-N step objective, 66-feature observations, expanded step action limits, and 5 N initial curriculum remain. Fresh SAC launchers also select the corrected world. Legacy SAC resume retains the saved world rather than silently replacing its dynamics.

Previous TD3 run remains stopped at 25001 steps with weights/replay preserved. Previous training worlds, diagnostic copies and historical run artifacts remain unchanged. No long training session was started when applying this correction.

Evidence: runs/leg_joint_verification_20261024_candidate_neutral/verification.md and runs/stepping_v3_validation_20261024_01_corrected_model/validation.json. Rebuild helper: correct_leg_model.py. Existing policies/replay describe the old kinematics and should not be treated as a valid corrected-model continuation.
