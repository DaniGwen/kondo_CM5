# Kondo TD3 shoulder filter patch (v3)

## Joint diagnostics

### Reset/contact initialization finding

Controlled tests identified static hip saturation caused by the initialization
path: driving motors from the saved bent pose into zero stance after ground
contacts form leaves both hip-roll motors at their torque limits. In three
neutral repetitions, resetting all 22 actual joint coordinates and motor targets
before settling reduced hip torque to about 12%, at the same effective friction.
Zeroing all motor targets alone did not solve it. Official R2025a joint/mass
parameters and reconstructed zero configurations match the expanded active world.

The server now supports opt-in `KONDO_RESET_JOINTS=1` for that clean reset.
Set it in the environment that launches Webots, then restart the controller.
With the setting absent, reset behavior remains compatible with existing runs.
Validate policies again before adopting it; a new training experiment should
discard old replay because episode initialization changed. The separate
`KONDO_RESET_ALL_TARGETS=1` switch exists only as a diagnostic control.

`test_stance_sweep.py` supports `--reset-joints`, `--reset-all-targets`,
`--friction`, `--gravity` and `--ground-z` for isolated ablations. Explicit friction
uses a `default`/`default` material pair in the diagnostic world. These flags
do not modify production contact labels or world settings.

`audit_twin_model.py <official-R2025a-Khr3hv.proto> --output <report.json>` compares
joint axes/anchors/motor limits, endpoint translations, reconstructed zero-pose
rotations, and Physics blocks with the expanded world. This is a consistency
audit against Webots, not a hardware-calibration certificate.

`python test_stance_sweep.py` performs a separate static neutral-stance sweep at
mirrored hip offsets -4/-2/0/+2/+4 degrees, three repetitions each. It records
eight seconds per trial and scores the final two seconds, without applying
pushes. Robot-wide contact points are attributed to the foot/ankle Solid IDs;
`contacts.csv` records foot poses and whole-robot center of mass. `summary.csv`
contains tracking error, torque usage and contact-presence fractions. Contact
presence is not a measurement of ground reaction force. The production world,
model files, and action/torque limits are preserved. Custom `--offsets`,
`--repetitions`, `--output-dir`, `--port`, and `--webots-port` are supported.

`python diagnose_joints.py` launches a separate instrumented world and compares
the saved SAC actor, preserved TD3 reference and neutral controller under the
same 4/6/8 N, 160 ms pushes. It adds position sensors to a new sibling world file;
the original world and model files are preserved. The server's optional
`KONDO_JOINT_DIAGNOSTICS` setting writes measured joint angles, server targets,
and estimated motor torque. Normal training omits this setting and keeps the
existing eight-double state protocol.

The client records requested/filtered targets and joins samples by episode,
step and joint into `joint_traces.csv`. `joint_summary.csv` includes displacement,
tracking error and time near action/torque limits. `outcomes.csv` verifies trial
behavior against earlier evaluations. Outputs default to a new timestamped run
directory; `--output-dir`, `--port`, and `--webots-port` select alternatives.
Use `python plot_joint_diagnostics.py <output-directory>` to plot the 6 N,
0-degree hip/knee traces. These are diagnostic evaluations, not training runs.

## SAC training

### Clean-reset SAC experiment

`python launch_sac_clean_reset.py --run-dir ./runs/<new-run-name> --steps 500000`
starts a new SAC policy and fresh replay in two separate minimized Webots
instances. Both use `KONDO_RESET_JOINTS=1`. Training uses the mixed-push
curriculum (60% 0.5–4 N, 40% 4 N to current maximum, 160 ms). Current maximum
begins at 4 N and can advance through 5 N to 6 N after two evaluation gates pass.

The trainer evaluates deterministically every 25,000 steps and at startup/end,
and writes `best_model.zip`, `reference_model.zip`, per-trial CSVs and
`evaluations/history.jsonl`. Reference preservation in this fresh experiment is
relative to the initial SAC policy; it is not a guarantee of matching the older
TD3 policy. The fixed 4/6/8 N suite ranks checkpoints. Final conclusions still
require testing new disturbance directions and starting states.

`experiment.json` records settings and `source_snapshot/` preserves the code.
`pretraining_*.csv` preserves the clean-reset evaluation of the old policies.
`launch_status.json` tracks the launcher, while `sac_status.json` tracks actual
training. The usual `STOP` file requests save-and-stop. Keep both simulation
windows open; a disconnected evaluator can be restarted once automatically.
Default controller ports are 5582/5583 and application ports 1264/1265.
Existing actor files, replay buffers, and production world defaults are not
overwritten. Joint limits, reward, observations and motor torque remain unchanged.

The new launcher is for fresh runs. For any manual resume, use the SAC trainer's
`--resume` mode with its saved replay and restart both correctly configured
servers, preserving the run's original clean reset and curriculum settings.

`train_kondo_balance_sac.py` trains a new Stable-Baselines3 SAC `MlpPolicy` from
scratch. It shares `KondoBalanceEnv` with TD3, so the 6 observations, 14 actions,
leg limits of +/-0.10 rad, shoulder limits of +/-0.20 rad, reward, and filters
remain identical. Switching algorithms alone does not expand hip/knee movement
or add joint feedback. SAC uses its stochastic policy with automatic entropy
tuning; it does not reuse TD3 weights or add the TD3 normal-action-noise object.

Restart the Webots TCP controller before connecting. From this directory:

```powershell
python train_kondo_balance_sac.py --run-dir ./runs/sac_balance_01 --steps 500000
```

The run directory must be new. Defaults are CPU, learning rate 0.0003, batch size
256, replay capacity 1,000,000, 5,000 warmup steps, and a checkpoint every 25,000
steps. Select `--host` and `--port` to use a different controller. Disturbances
come from that Webots controller's configuration. This standalone trainer does
not start Webots. The clean-reset launcher sets `KONDO_EVAL_PORT` to enable
the optional independent periodic evaluation callback.
Use an ordinary controller without `KONDO_CURRICULUM_FILE` for standalone SAC.
Do not connect it to the simulator already serving an active training client.

Outputs are `kondo_sac.zip`, `kondo_sac_replay_buffer.pkl`, `checkpoints_sac/`,
`tensorboard_sac/`, `sac_run.json`, and `sac_status.json`. Ctrl+C or a `STOP` file
requests saving. Resume explicitly after restarting the controller:

```powershell
python train_kondo_balance_sac.py --run-dir ./runs/sac_balance_01 --resume --steps 500000
```

On resume, `--steps` means additional steps for that invocation. Use
`--reset-replay` only when deliberately discarding replay; gradient updates wait
until a missing/short buffer has enough fresh transitions. Replay-load failures
are reported rather than silently ignored. A SAC manifest is required to resume,
and existing run directories are rejected without `--resume`.

Evaluate the saved SAC policy deterministically with the same matched tests:

```powershell
python evaluate_kondo_balance.py --algorithm sac --mode compare --model ./runs/sac_balance_01/kondo_sac.zip --episodes 3 --forces 4 6 8 --durations 160 --angles 0 90 180 270 --output balance_comparison_sac.csv
```

The comparison CSV labels the policy as `sac` and keeps neutral baseline rows.
These changes do not stop or replace any running TD3 experiment.

## Push recovery comparison

### Isolated fine-tuning after the comparison

#### Mixed pushes with evaluation gates

Use `launch_finetune_balance.py --adaptive --source-model <preserved-original.zip>`
for the revised experiment. Start a fresh run directory; this copies the source
actor and starts with no replay. Learning rate is 0.0001 and exploration sigma is
0.10 in SB3's normalized action space. The reward, observations, filters, and
action limits remain the same.

Every push lasts 160 ms. At every stage, 60% of pushes sample 0.5–4 N; 40% sample
4 N up to the current harder maximum. The harder maximum begins at 4 N and can
advance to 5 N, then 6 N. Each episode has a two-second settling period. The
controller reads the atomically updated `curriculum.json` at episode resets.

A separate Webots instance evaluates deterministic policy actions every 25,000
additional steps, plus before training and at the end. The fixed selection suite
uses 4/6/8 N, four cardinal directions, and the existing recovery criteria.
The evaluation also tests 5 N when that is the current stage. Training and
evaluation use different TCP connections and simulator instances, so evaluation
does not insert transitions or resets into the training replay buffer.

If the dedicated evaluation simulator disconnects, the callback closes the old
socket, starts a replacement evaluation instance, and retries the complete test
suite once. It does not modify the training simulator or replay. If replacement
fails, training exits through its normal save path and reports the failure.
Use `--resume --adaptive --run-dir <existing-run>` with free application/controller
ports to continue a saved run: its replay, original evaluation baseline, best
score, and curriculum stage are retained. `--steps` remains the total additional
training target relative to the original model, rather than steps to add on every
resume. Previous logs and revised sources are archived in a resume snapshot.

Advancement requires two consecutive checks that preserve every successful
reference scenario and pass at least three of four directions at the current
stage, including at least one measurable recovery. An outcome below the tilt
threshold counts as disturbance resistance rather than a recovery. A failed
check resets the advancement streak. Stages never increase merely with episode
count. A passed stage remains in place after later regressions; the gate does
not roll back network weights or reduce difficulty automatically.

`best_model.zip` starts as the original policy. A candidate can replace it only
if it preserves all successful reference cases and improves the fixed suite's
lexicographic score: fewer failed trials, then fewer falls, then lower mean peak
tilt rounded to 0.1 degrees. A timeout counts as failed even without a fall.
`kondo_td3_v2.zip` remains the latest training model, not necessarily the best.
Individual evaluation CSVs and `evaluations/history.jsonl` record each decision.
These fixed cases select checkpoints; a selected model still needs confirmation
on additional directions, forces, and starting states before claiming general
robustness.

Use `--webots-port` to choose the training instance's application port; the
evaluation instance uses the next port. `--port` chooses the training controller
TCP port and `--eval-port` chooses the evaluation controller TCP port. The
`STOP` file still requests a normal save-and-stop. Do not start a second trainer
while this launcher is already running.

`python launch_finetune_balance.py` creates a new timestamped directory under
`runs/`, preserves the evaluated actor as `reference_model.zip`, and initializes
the training actor from a separate copy. No replay is copied. The manifest records
the source model hash and all experiment settings; source snapshots and the two
evaluation CSVs are also retained.

The launcher starts a separate minimized Webots instance (application port 1235,
controller TCP port 5556) and runs 500,000 additional steps. Pushes last 160 ms,
sample forces from 4 N to a maximum ramping from 4 to 6 N over 30 episodes, and
use uniformly random horizontal directions. The first three episodes have no
pushes, and each episode has 125 settling steps. There is a two-second cooldown
between push onsets, subject to completion of the previous recovery trial.
Training retains the existing observation, action, filter, and reward definitions.

Use `--steps`, `--port`, `--source-model`, and `--run-dir` to change launch settings.
An existing run directory is rejected to prevent overwriting another experiment.
The launcher requires the Python environment used for training and a local Webots
installation. It waits for the controller log before starting the training client.
Inspect `training.log`, `controller.log`, `webots.log`, and `status.json` inside the
new run directory. The controller writes directly to `controller.log`; this also
provides startup readiness on Windows when Webots does not forward stdout.
Checkpoints are saved every 50,000 additional steps. To stop with model and replay
saving, create an empty file named `STOP` inside the run directory. Delete that
file before any later resume. The launcher leaves its simulation window available
after training so it can be closed normally.

After stopping or completing this run, restart the evaluation controller and run
the same 4/6/8 N comparison with the new run's model, saving to a new CSV. Compare
against the original saved comparison, especially 0/180 degree falls and the
270 degree tilt/recovery time. This run changes training exposure; improvement
must still be established by evaluation.

The updated server reports `outcome=no_measurable_disturbance`, `recovered`, or
`failed` for each push. No measurable disturbance is excluded from the recovery
success denominator. Episode survival is separately logged as
`balance/standing_success_rate_last_50`; the older success metric remains as an
alias for compatibility. Recovery outcomes appear in the server console, not in
the training client's TensorBoard stream (the eight-double state is unchanged).

Training pushes now default to 5 simulation steps (80 ms), with force ramping
from a maximum of 1 N to 4 N. The force is applied on every push step. Server-side
environment settings are `KONDO_PUSH_DURATION_STEPS`,
`KONDO_PUSH_MAX_FORCE_INITIAL`, and `KONDO_PUSH_MAX_FORCE_FINAL`. Set these before
launching Webots so its controller inherits them. Use duration 1 and force maxima
4/20 to reproduce the previous disturbance settings. Keep duration shorter than
`KONDO_RECOVERY_TIMEOUT_STEPS` (default 125). Logs include actual duration and impulse.

Stop training and restart the updated Webots controller before evaluation. From
this directory, run a small paired comparison first:

```powershell
python evaluate_kondo_balance.py --mode compare --model ./kondo_td3_v2.zip --episodes 1 --forces 1 2 4 --durations 80 160 --angles 0 90 180 270 --output balance_comparison.csv
```

Choose an existing checkpoint path if the model ZIP is elsewhere. This command
runs 48 trials: each policy/baseline pair gets one identical scheduled world-frame
push after 125 steps (2 seconds). Random curriculum pushes are bypassed. Robot
and descendant joint states are restored before each trial; the baseline sends
zero offsets, and TD3 inference is deterministic. Repeat with `--episodes 3` for
three repetitions. This tests the current actor without training or saving it.

The CSV records outcome, fall status, peak tilt, time from push onset to outcome,
force, angle, duration, impulse, and mean command norm. Trials end at recovery,
fall, or timeout. A fall before the push is explicitly marked `pre_push_fall`
and excluded from the measurable-disturbance recovery rate. The default recovery
threshold is 2 degrees, with 10 stable samples and a 125-step observation window.
The CLI threshold/window options affect CSV classification; server console
classification uses its own `KONDO_RECOVERY_*` settings. Keep them equal when
comparing the console with CSV. Requested durations round up to 16 ms multiples;
the server also requires them to be shorter than its recovery timeout.

Use the updated client and server together: the comparison uses an optional
reset-packet extension, while regular training retains the existing packet format.
An old server cannot execute the scheduled trials. Restart Webots from the same
world for subsequent comparison runs. Differences in contact physics can still
introduce small variation; this is a matched disturbance test, not a guarantee
of bit-identical trajectories.

For continued training, create a new run directory, copy in the actor if desired,
and set `KONDO_RESET_REPLAY=1` because the disturbance dynamics changed. Existing
model and replay files are not modified by the comparison.

## Training improvements

Exploration now uses `KONDO_NOISE_SIGMA` (default `0.20`) in SB3's normalized action space. This corresponds to standard deviations of 0.02 rad for legs and 0.04 rad for shoulders before EMA, limits and shoulder filtering. Previously the action limits were applied twice, greatly reducing exploration.

Resuming with a missing, discarded or short replay buffer collects enough fresh transitions to reach `KONDO_WARMUP_STEPS` (default 5000) before gradient updates. Episode metrics retain only the latest 50 episodes. Non-finite actions are rejected before transmission.

Use a separate run directory to compare against existing checkpoints without overwriting them. In PowerShell, from the directory containing the training script:

```powershell
$env:KONDO_RUN_DIR = './runs/noise_fix_01'
$env:KONDO_TOTAL_TIMESTEPS = '2000000'
python train_kondo_balance.py
```

To fine-tune an existing actor, copy its `kondo_td3_v2.zip` into the new run directory. Set `$env:KONDO_RESET_REPLAY = '1'` to discard replay generated with different filter dynamics; the run's replay file is replaced on normal shutdown. With no copied model, the run starts from scratch. The default run directory remains the current directory for compatibility.

Compare the new run and the original with `evaluate_kondo_balance.py --model <path>` and `--mode baseline`. Restart the Webots controller with the same disturbance settings and seed before each comparison: client environment seeds do not reset the server's disturbance RNG or curriculum. Reward and policy dimensions are unchanged; improved recovery still requires simulation evaluation.

These files are based on the previously adjusted v2 scripts. The ONNX exporter is unchanged; the Webots server now supports timed pushes and paired evaluation. The shared `shoulder_filter.py` is imported by both `train_kondo_balance.py` and `run_RL_balance.py` and **must be placed alongside each script** (copy it into both script directories if those directories differ).

## Behavior

Only policy joint indices 12 and 13 (RightShoulder, LeftShoulder) are filtered. First, absolute shoulder offsets smaller than 0.008 radians (~0.46 degrees) are set to zero. Next, a shoulder target is held at the last sent target if the difference is smaller than 0.008 radians. Other joints retain the existing EMA and rate limits. Both filters operate on **joint offset radians**, not absolute servo angles. Strict `<` comparisons mean a change equal to the threshold is transmitted.

The filter is applied after EMA and rate limiting, before sending to Webots or to physical servos. Its state is the previous applied command. The Gym environment resets this state at each episode. The physical runner resets state at process startup.

## Compatibility and caution

This does **not** require retraining to run a deterministic test with the existing v2 actor, since action and observation dimensions are unchanged. However, the filter changes the environment's action dynamics, so continued training or evaluation with the filter may behave differently from the previously trained policy. For rigorous training comparisons, back up your model and replay buffer first. Avoid mixing old transitions with new filter dynamics: move the old replay buffer aside before fine-tuning, or use a fresh experiment. Do not claim the previous training statistics are representative of the modified controller.

Physical control remains **disabled by default** and requires independently verified unique servo IDs, standing angles, sensor mapping, joint signs, and a supported test fixture. The filter reduces small command changes; it does not guarantee prevention of mechanical wear or eliminate physical servo hunting caused by hardware control loops. The runner still calls `set_angles` each cycle when enabled, even if shoulder targets are held. Whether a held target produces bus traffic depends on the API.

## Installation

1. Stop the training and Webots controller; back up your existing files, model ZIP and replay buffer.
2. Replace `train_kondo_balance.py` in your training directory and put `shoulder_filter.py` beside it.
3. Replace `run_RL_balance.py` in the robot runner directory and also put `shoulder_filter.py` beside it.
4. Use the updated `webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py`. Restart Webots and then the training script or evaluator.
5. First compare deterministic evaluations with the old and new filter, ideally using the same push seeds and a neutral-controller baseline. Measure shoulder command changes and push recovery.

Change both thresholds in the shared file consistently in both directories if you tune them. Start at 0.008 rad; this is a test value, not a hardware-certified safe threshold.
