# Local training dashboard

Double-click `Start Training UI.cmd`. It reuses a running dashboard or starts one with the installed training Python, then opens your default browser. Alternatively run `python training_ui.py` from this folder and open http://127.0.0.1:8765. The service binds only to localhost. Closing the page or dashboard does not stop training.

Choose TD3 or SAC, a unique run name, total steps and learning rate. TD3 defaults to 0.0001; SAC to 0.0003. Reward defaults: torque 0.75, saturation 0.75, tracking 0.25, recovery landing 0.75. Values are nonnegative weights; zero disables a component. Survival, uprightness, smoothness and fall rewards remain fixed. Reward changes apply only to new runs; they never edit the running session.

Every new run starts with fresh weights/replay, corrected axes/neutral pose, box slab, friction 1.0, +25% leg torque, 66 observations, stepping actions and above-4-N landing objective. The curriculum starts at 5 N. It uses an independent evaluation simulator and saves checkpoints/evaluations every 25000 steps. Each evaluation includes a no-push load gate. No successful stepping is guaranteed by the reward alone.

The Start button is disabled while the dedicated training/evaluation ports are occupied. Close finished Webots instances before launching another session. Stop and save creates a STOP file for the selected active run; current evaluation may finish before model and replay are saved. It does not force-kill processes.

config.json stores all chosen values; experiment.json stores the environment and world version. Each run records a model/world source snapshot. final_model.zip, replay.pkl, best_model.zip, checkpoints/, training_status.json, training.log and evaluations/ are saved in runs/<name>. The UI is a fresh-run launcher; it does not resume old runs or convert incompatible models. Status reads the existing corrected TD3 run without controlling it. A stale state file without active ports is not proof that a trainer is alive.

Documentation buttons show model corrections, observation layout, stepping details and reusable robot instructions. For new worlds use CorrectedKondoKHR3HV.proto, not the stock Kondo. See CORRECTED_ROBOT.md.
