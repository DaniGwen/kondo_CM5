"""Train TD3 balance policy against kondo_tcp_server.py in Webots.

Protocol:
  client -> Webots: 14 little-endian doubles (joint offset actions in radians)
  reset: 14 values of 999.0
  Webots -> client: 8 doubles = observation[6], tilt_rad, root_height_m

The six policy features are kept compatible with run_RL_balance.py:
  [forward_accel, lateral_accel, 9.81, roll_rate, pitch_rate, 0.0]

This version deliberately writes a new model name. Do not resume the old
kondo_td3_smooth.zip replay buffer: its transitions were generated with a
misaligned request/response protocol and a different reward.
"""

from __future__ import annotations

import os
from collections import deque
import socket
import struct
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from shoulder_filter import filter_shoulders, SHOULDER_NEUTRAL_DEADBAND_RAD, SHOULDER_CHANGE_THRESHOLD_RAD

HOST = os.getenv("KONDO_WEBOTS_HOST", "127.0.0.1")
PORT = int(os.getenv("KONDO_WEBOTS_PORT", "5555"))
TIMEOUT_SECONDS = float(os.getenv("KONDO_SOCKET_TIMEOUT", "30"))
ACTION_COUNT = 14
STEPPING = os.getenv('KONDO_OBSERVATION_VERSION', '1') == '3'
EXTENDED_OBSERVATION = os.getenv('KONDO_OBSERVATION_VERSION', '1') in ('2', '3')
OBSERVATION_COUNT = 66 if STEPPING else 51 if EXTENDED_OBSERVATION else 6
STATE_COUNT = OBSERVATION_COUNT + 2
RESET_SENTINEL = 999.0
MAX_EPISODE_STEPS = 4000

ACTION_LIMITS = np.array([0.10] * 12 + [0.20, 0.20], dtype=np.float32)
if STEPPING:
    ACTION_LIMITS = np.array([.1, .15, .25, .4, .2, .15] * 2 + [.2, .2], dtype=np.float32)
SMOOTHING_ALPHA = 0.25
# Per-simulation-step limit, radians. This same limit is used by the real runner.
ACTION_DELTA_LIMITS = np.array([0.03] * 12 + [0.04, 0.04], dtype=np.float32)

# These thresholds are model-specific starting points. Check body height and
# tilt telemetry in Webots before trusting them for success/fall classification.
FALL_TILT_RAD = float(os.getenv("KONDO_FALL_TILT_RAD", "0.85"))
FALL_HEIGHT_M = float(os.getenv("KONDO_FALL_HEIGHT_M", "0.13"))


class KondoBalanceEnv(gym.Env):
    """One Webots instance controlled synchronously over TCP."""

    metadata = {"render_modes": []}

    def __init__(self, host: str = HOST, port: int = PORT):
        super().__init__()
        self.current_step = 0
        if STEPPING:
            from stepping_reward import RecoveryStepReward
            self.step_reward = RecoveryStepReward()
        self.fall_counter = 0
        self.previous_raw_action = np.zeros(ACTION_COUNT, dtype=np.float32)
        self.previous_applied_action = np.zeros(ACTION_COUNT, dtype=np.float32)
        self.last_tilt = 0.0
        self.last_height = 0.24
        self.max_tilt_in_episode = 0.0
        self.action_space = spaces.Box(
            low=-ACTION_LIMITS,
            high=ACTION_LIMITS,
            dtype=np.float32,
        )
        # The two constant features are intentionally included in the broad box;
        # the true runtime values are validated separately.
        self.observation_space = spaces.Box(
            low=np.full(OBSERVATION_COUNT, -np.inf if EXTENDED_OBSERVATION else -100., dtype=np.float32),
            high=np.full(OBSERVATION_COUNT, np.inf if EXTENDED_OBSERVATION else 100., dtype=np.float32),
            dtype=np.float32,
        )

        self.sock = socket.create_connection((host, port), timeout=TIMEOUT_SECONDS)
        self.sock.settimeout(TIMEOUT_SECONDS)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._closed = False

    @staticmethod
    def _recv_exact(sock: socket.socket, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            data = sock.recv(count - len(chunks))
            if not data:
                raise ConnectionError("Webots server disconnected while sending a state packet.")
            chunks.extend(data)
        return bytes(chunks)

    def _receive_state(self) -> Tuple[np.ndarray, float, float]:
        data = self._recv_exact(self.sock, STATE_COUNT * 8)
        values = struct.unpack('<' + str(STATE_COUNT) + 'd', data)
        obs = np.asarray(values[:OBSERVATION_COUNT], dtype=np.float32)
        tilt = float(values[OBSERVATION_COUNT])
        height = float(values[OBSERVATION_COUNT + 1])
        if not np.all(np.isfinite(obs)) or not np.isfinite(tilt) or not np.isfinite(height):
            raise RuntimeError("Webots returned non-finite observation/telemetry.")
        return obs, tilt, height

    def _send_action(self, action: np.ndarray) -> None:
        packet = struct.pack("<14d", *[float(x) for x in action])
        self.sock.sendall(packet)

    def _filter_action(self, raw_action: np.ndarray) -> np.ndarray:
        bounded = np.clip(raw_action, -ACTION_LIMITS, ACTION_LIMITS).astype(np.float32)
        smoothed = SMOOTHING_ALPHA * bounded + (1.0 - SMOOTHING_ALPHA) * self.previous_applied_action
        delta = np.clip(
            smoothed - self.previous_applied_action,
            -ACTION_DELTA_LIMITS,
            ACTION_DELTA_LIMITS,
        )
        applied = (self.previous_applied_action + delta).astype(np.float32)
        # Shoulder-only deadband and last-sent-target hysteresis, matching hardware.
        applied = filter_shoulders(applied, self.previous_applied_action)
        self.previous_applied_action = applied.copy()
        return applied

    def step(self, action):
        if self._closed:
            raise RuntimeError("Environment is closed.")

        self.current_step += 1
        raw_action = np.asarray(action, dtype=np.float32).reshape(ACTION_COUNT)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("Action must contain only finite values.")
        raw_action = np.clip(raw_action, -ACTION_LIMITS, ACTION_LIMITS)
        previous_applied_action = self.previous_applied_action.copy()
        applied_action = self._filter_action(raw_action)
        self._send_action(applied_action)
        obs, tilt, height = self._receive_state()

        self.last_tilt = tilt
        self.last_height = height
        self.max_tilt_in_episode = max(self.max_tilt_in_episode, abs(tilt))

        action_scale = np.maximum(ACTION_LIMITS, 1e-6)
        normalized_action = applied_action / action_scale
        previous_normalized = previous_applied_action / action_scale
        # Reward the achieved upright posture, survival, and quiet commands—not
        # arbitrary limb movement conditional on an accelerometer threshold.
        upright_reward = 2.0 * float(np.exp(-((tilt / 0.30) ** 2)))
        angular_rate = np.clip(obs[3:5], -10.0, 10.0)
        angular_rate_penalty = 0.02 * float(np.mean(np.square(angular_rate / 5.0)))
        effort_penalty = 0.05 * float(np.mean(np.square(normalized_action)))
        action_delta = normalized_action - previous_normalized
        smoothness_penalty = 0.03 * float(np.mean(np.square(action_delta)))
        reward = 1.0 + upright_reward - angular_rate_penalty - effort_penalty - smoothness_penalty
        if EXTENDED_OBSERVATION:
            from balance_reward_v2 import revised_reward
            reward = revised_reward(reward, obs)
        step_bonus = 0.
        if STEPPING:
            step_bonus = self.step_reward.update(obs, tilt, fell=abs(tilt) >= FALL_TILT_RAD or height <= FALL_HEIGHT_M)
            reward += step_bonus

        fell = (abs(tilt) >= FALL_TILT_RAD) or (height <= FALL_HEIGHT_M)
        terminated = bool(fell)
        truncated = bool(self.current_step >= MAX_EPISODE_STEPS and not terminated)
        if terminated:
            reward -= 20.0
        if truncated:
            reward += 10.0

        info = {
            "tilt_rad": float(tilt),
            "height_m": float(height),
            "max_tilt_rad": float(self.max_tilt_in_episode),
            "episode_steps": int(self.current_step),
            "is_success": bool(truncated and not terminated),
            "standing_success": bool(truncated and not terminated),
            "fell": bool(fell),
        }
        if EXTENDED_OBSERVATION:
            info.update(hip_torque_fraction=float(np.mean(np.abs(obs[[24, 30]]))),
                        leg_torque_fraction=float(np.mean(np.abs(obs[23:35]))),
                        leg_tracking_error=float(np.mean(np.abs(obs[37:49]))))
        if STEPPING:
            info['recovery_step_bonus'] = step_bonus
            info['recovery_landings'] = self.step_reward.last_landings
        self.previous_raw_action = raw_action.copy()
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if STEPPING:
            self.step_reward.reset()
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment.")
        self.current_step = 0
        self.fall_counter = 0
        self.previous_raw_action.fill(0.0)
        self.previous_applied_action.fill(0.0)
        self.max_tilt_in_episode = 0.0

        reset_values = [RESET_SENTINEL] * ACTION_COUNT
        if options and "push_trial" in options:
            force, angle, duration, start = options["push_trial"]
            if (not np.all(np.isfinite([force, angle, duration, start])) or
                    not 0 <= force <= 100 or duration < 1 or start < 1 or
                    duration != int(duration) or start != int(start)):
                raise ValueError("Invalid push_trial")
            reset_values[1:6] = [1.0, force, angle, duration, start]
        reset_packet = struct.pack("<14d", *reset_values)
        self.sock.sendall(reset_packet)
        obs, tilt, height = self._receive_state()
        if options and "push_trial" in options:
            if obs[5] != -999.0:
                raise RuntimeError("Server lacks scheduled-push support; restart the updated Webots controller.")
            obs[5] = 0.0
        self.last_tilt = tilt
        self.last_height = height
        return obs, {"tilt_rad": tilt, "height_m": height}

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
        super().close()


class BalanceMetricsCallback(BaseCallback):
    """Log episode performance; this does not require a second Webots server."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.successes = deque(maxlen=50)
        self.lengths = deque(maxlen=50)
        self.falls = deque(maxlen=50)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for info in infos:
            for name in ('hip_torque_fraction', 'leg_torque_fraction', 'leg_tracking_error', 'recovery_step_bonus'):
                if name in info:
                    self.logger.record('balance/' + name, info[name])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.successes.append(float(info.get("is_success", False)))
            self.lengths.append(float(info.get("episode_steps", 0)))
            self.falls.append(float(info.get("fell", False)))
            self.logger.record("balance/success_rate_last_50", float(np.mean(self.successes)))
            self.logger.record("balance/standing_success_rate_last_50", float(np.mean(self.successes)))
            self.logger.record("balance/episode_steps_last_50", float(np.mean(self.lengths)))
            self.logger.record("balance/fall_rate_last_50", float(np.mean(self.falls)))
            self.logger.record("balance/max_tilt_rad", float(info.get("max_tilt_rad", 0.0)))
        return True


class StopFileCallback(BaseCallback):
    """Allow a background run to stop through the normal model-saving path."""

    def __init__(self, run_dir):
        super().__init__()
        self.stop_file = Path(run_dir) / "STOP"

    def _on_step(self):
        return not self.stop_file.exists()


def main():
    seed = int(os.getenv("KONDO_TRAIN_SEED", "123"))
    total_timesteps = int(os.getenv("KONDO_TOTAL_TIMESTEPS", "2000000"))
    warmup_steps = int(os.getenv("KONDO_WARMUP_STEPS", "5000"))
    noise_sigma = float(os.getenv("KONDO_NOISE_SIGMA", "0.20"))
    if total_timesteps <= 0 or warmup_steps < 1:
        raise ValueError("Total timesteps and warmup steps must be positive.")
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("KONDO_NOISE_SIGMA must be finite and nonnegative.")
    run_dir = Path(os.getenv("KONDO_RUN_DIR", "."))
    checkpoint_dir = run_dir / "checkpoints_kondo_v2"
    tensorboard_dir = run_dir / "kondo_tensorboard_v2"
    np.random.seed(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = KondoBalanceEnv()
    model_path = run_dir / "kondo_td3_v2.zip"
    replay_path = run_dir / "kondo_td3_v2_replay_buffer.pkl"
    action_noise = NormalActionNoise(
        mean=np.zeros(ACTION_COUNT, dtype=np.float32),
        # SB3 adds noise in normalized [-1, 1] action space, then unscales.
        sigma=np.full(ACTION_COUNT, noise_sigma, dtype=np.float32),
    )

    print(f"Shoulder filter: neutral deadband={SHOULDER_NEUTRAL_DEADBAND_RAD:.4f} rad, "
          f"command threshold={SHOULDER_CHANGE_THRESHOLD_RAD:.4f} rad", flush=True)

    if model_path.exists():
        print(f"Loading {model_path}; this is the v2 model, not the old smooth model.", flush=True)
        model = TD3.load(
            str(model_path), env=env, action_noise=action_noise,
            tensorboard_log=str(tensorboard_dir), device="auto",
            **({"learning_rate": float(os.environ["KONDO_LEARNING_RATE"])}
               if "KONDO_LEARNING_RATE" in os.environ else {}),
        )
        if replay_path.exists() and os.getenv("KONDO_RESET_REPLAY", "0") != "1":
            try:
                model.load_replay_buffer(str(replay_path))
                print("Loaded v2 replay buffer.", flush=True)
            except Exception as exc:
                print(f"Warning: could not load replay buffer ({exc}); continuing with a fresh buffer.", flush=True)
        else:
            print("No v2 replay buffer found; continuing with a fresh buffer.", flush=True)
    else:
        print("Creating new TD3 v2 model.", flush=True)
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=warmup_steps,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_delay=2,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            seed=seed,
            device="auto",
        )

    # learning_starts uses the lifetime timestep counter when resuming.
    # Refill a missing/short buffer before allowing gradient updates.
    replay_size = model.replay_buffer.size()
    missing_steps = max(0, warmup_steps - replay_size)
    if missing_steps:
        model.learning_starts = max(model.learning_starts, model.num_timesteps + missing_steps)
        print(f"Collecting {missing_steps} warm-up steps before updates.", flush=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix="kondo_td3_v2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = BalanceMetricsCallback()
    callbacks = CallbackList([checkpoint_callback, metrics_callback, StopFileCallback(run_dir)])
    if os.getenv("KONDO_EVAL_PORT"):
        from adaptive_evaluation import AdaptiveEvaluationCallback
        from restart_evaluation import restart_evaluation
        evaluation_env = KondoBalanceEnv(port=int(os.environ["KONDO_EVAL_PORT"]))
        callbacks.callbacks.append(AdaptiveEvaluationCallback(
            evaluation_env, run_dir,
            restart=lambda: restart_evaluation(run_dir, int(os.environ["KONDO_EVAL_PORT"]))))

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving current progress.", flush=True)
    finally:
        try:
            model.save(str(model_path))
            try:
                model.save_replay_buffer(str(replay_path))
            except Exception as exc:
                print(f"Warning: replay buffer save failed: {exc}", flush=True)
        finally:
            env.close()
        print("Saved kondo_td3_v2.zip and replay buffer (if supported).", flush=True)


if __name__ == "__main__":
    main()
