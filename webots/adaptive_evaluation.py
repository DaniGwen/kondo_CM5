"""Periodic deterministic checks on a separate Webots environment."""
import csv
import json
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from recovery_metrics import RecoveryTracker
from adaptive_curriculum import rank, preserves_reference, can_advance


def evaluate_trials(model, env, forces):
    rows = []
    for force in forces:
        for angle in (0, 90, 180, 270):
            obs, _ = env.reset(options={'push_trial': (force, angle, 10, 125)})
            tracker = RecoveryTracker()
            landing_bonus = 0.
            for step in range(1, 250):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                landing_bonus += info.get('recovery_landings', 0)
                if step >= 125:
                    age = step - 124
                    tracker.update(float(np.degrees(info['tilt_rad'])), age,
                                   pushing=age < 10, fell=info['fell'])
                if terminated or truncated or tracker.outcome is not None:
                    break
            rows.append(dict(force_N=force, angle_deg=angle,
                             outcome=tracker.outcome or 'failed', fell=info['fell'],
                             peak_tilt_deg=tracker.peak, outcome_time_s=tracker.elapsed_steps * .016,
                             pre_push_fall=step < 125))
            if getattr(env.observation_space, 'shape', ()) == (66,):
                rows[-1]['rewarded_recovery_landings'] = int(landing_bonus)
    return rows


class AdaptiveEvaluationCallback(BaseCallback):
    def __init__(self, env, run_dir, interval=25000, restart=None):
        super().__init__()
        self.eval_env = env
        self.run = Path(run_dir)
        self.interval = interval
        self.stage = 0
        self.gate_streak = 0
        self.reference = None
        self.best_rank = None
        self.last_eval = None
        self.restart = restart

    def write_curriculum(self):
        path = self.run / 'curriculum.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(stage=self.stage, max_force_N=[4, 5, 6][self.stage],
                                             gate_streak=self.gate_streak)), encoding='utf-8')
        temporary.replace(path)

    def evaluate(self, initial=False):
        gate_force = [4, 5, 6][self.stage]
        forces = sorted(set([4, 6, 8, gate_force]))
        was_training = self.model.policy.training
        hold = None
        try:
            try:
                rows = evaluate_trials(self.model, self.eval_env, forces)
            except OSError:
                if self.restart is None:
                    raise
                print('[EVALUATION] Connection lost; restarting evaluation simulator and retrying full suite', flush=True)
                self.eval_env.close()
                self.eval_env = self.restart()
                rows = evaluate_trials(self.model, self.eval_env, forces)
            if getattr(getattr(self.eval_env, 'observation_space', None), 'shape', ()) in ((51,), (66,)):
                obs, _ = self.eval_env.reset(options={'push_trial': (0, 0, 10, 2000)})
                loads = []
                fell = False
                for step in range(625):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)
                    if step >= 500:
                        loads.append(float(np.mean(np.abs(obs[[24, 30]]))))
                    if terminated or truncated:
                        fell = info['fell']
                        break
                hold = dict(fell=fell, mean_hip_torque_fraction=float(np.mean(loads)) if loads else None)
        finally:
            self.model.policy.set_training_mode(was_training)
        fixed = [r for r in rows if r['force_N'] in (4, 6, 8)]
        if initial:
            self.reference = fixed
            self.best_rank = rank(fixed)
            self.model.save(str(self.run / 'best_model.zip'))
        qualifies = preserves_reference(self.reference, fixed)
        load_gate = hold is None or (not hold['fell'] and hold['mean_hip_torque_fraction'] is not None and hold['mean_hip_torque_fraction'] <= .6)
        promoted = not initial and qualifies and load_gate and rank(fixed) < self.best_rank
        if promoted:
            self.best_rank = rank(fixed)
            self.model.save(str(self.run / 'best_model.zip'))
        if not initial:
            gate = [r for r in rows if r['force_N'] == gate_force]
            self.gate_streak = self.gate_streak + 1 if load_gate and can_advance(self.reference, fixed, gate) else 0
            if self.gate_streak >= 2 and self.stage < 2:
                self.stage += 1
                self.gate_streak = 0
        self.write_curriculum()
        additional = self.num_timesteps - self.start_steps
        folder = self.run / 'evaluations'
        folder.mkdir(exist_ok=True)
        with (folder / f'trials_{additional:07d}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(additional_steps=additional, stage=self.stage,
                       max_training_force_N=[4, 5, 6][self.stage],
                       reference_preserved=qualifies, candidate_rank=rank(fixed),
                       best_rank=self.best_rank, saved_best=initial or promoted,
                       gate_streak=self.gate_streak)
        if hold is not None:
            summary.update(no_push_hold=hold, load_gate_passed=load_gate)
        with (folder / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(summary) + '\n')
        print('[EVALUATION] ' + json.dumps(summary), flush=True)
        self.last_eval = self.num_timesteps

    def _on_training_start(self):
        history = self.run / 'evaluations/history.jsonl'
        original = self.run / 'evaluations/trials_0000000.csv'
        if history.exists() and original.exists():
            import zipfile
            with zipfile.ZipFile(self.run / 'reference_model.zip') as archive:
                self.start_steps = json.loads(archive.read('data'))['num_timesteps']
            with original.open(newline='', encoding='utf-8') as stream:
                self.reference = [dict(force_N=float(r['force_N']), angle_deg=float(r['angle_deg']),
                                       outcome=r['outcome']) for r in csv.DictReader(stream)
                                  if float(r['force_N']) in (4, 6, 8)]
            last = json.loads(history.read_text(encoding='utf-8').splitlines()[-1])
            self.best_rank = tuple(last['best_rank'])
            self.stage = last['stage']
            self.gate_streak = last['gate_streak']
            self.evaluate()
        else:
            self.start_steps = self.model.num_timesteps
            self.evaluate(initial=True)

    def _on_step(self):
        if self.num_timesteps - self.last_eval >= self.interval:
            self.evaluate()
        return True

    def _on_training_end(self):
        try:
            if self.last_eval != self.num_timesteps:
                self.evaluate()
        finally:
            self.eval_env.close()
