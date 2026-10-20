import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import os
import struct
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

class KondoBalanceEnv(gym.Env):
    def __init__(self):
        super(KondoBalanceEnv, self).__init__()
        self.current_step = 0
        self.fall_counter = 0
        
        # Track both raw and smoothed actions for the EMA filter
        self.previous_raw_action = np.zeros(14, dtype=np.float32)
        self.previous_smoothed_action = np.zeros(14, dtype=np.float32)
        
        # 12 leg joints limited to +/- 0.1. 2 shoulder joints limited to +/- 0.2.
        low_bounds = np.array([-0.1]*12 + [-0.2, -0.2], dtype=np.float32)
        high_bounds = np.array([ 0.1]*12 + [ 0.2,  0.2], dtype=np.float32)
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(14,), dtype=np.float32)
        
        # Observation space: 3 Accel + 3 Gyro = 6 floats
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
        
        # Connect to Webots TCP Server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 5555))

    def step(self, action):
        self.current_step += 1
        
        # --- 1. ACTION FILTERING (The Anti-Twitch Engine) ---
        SMOOTHING_ALPHA = 0.25 
        smoothed_action = (SMOOTHING_ALPHA * action) + ((1.0 - SMOOTHING_ALPHA) * self.previous_smoothed_action)
        self.previous_smoothed_action = np.copy(smoothed_action)
        
        # Send the SMOOTHED action to Webots (14 doubles)
        action_data = struct.pack('14d', *smoothed_action)
        self.sock.send(action_data)
        
        # Receive new state from Webots (6 doubles = 48 bytes)
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        # Extract IMU data
        accel_x, accel_y, accel_z = state[0], state[1], state[2]
        gyro_roll, gyro_pitch = state[3], state[4]
        
        upright_reward = -2.0 * (abs(accel_x) + abs(accel_y))
        
        # Measure disturbance magnitude from IMU sensors
        disturbance = abs(accel_x) + abs(accel_y) + abs(gyro_roll) + abs(gyro_pitch)
        
        # --- 2. HARSH SHOULDER PENALTY ---
        # Shoulders are the last two joints in the 14-element array (indices 12 and 13)
        shoulder_movement = np.sum(np.abs(action))
        shoulder_penalty = -5.0 * shoulder_movement  # Heavily penalize arm flapping
        
        # --- 3. TARGETED LEG ARTICULATION ---
        # Right leg pitch: 2 (Hip), 3 (Knee), 4 (Ankle)
        # Left leg pitch: 8 (Hip), 9 (Knee), 10 (Ankle)
        leg_pitch_joints = [2, 3, 4, 8, 9, 10]
        leg_movement = np.sum(np.abs(action[leg_pitch_joints]))
        
        STABILITY_THRESHOLD = 0.4 
        
        if disturbance > STABILITY_THRESHOLD:
            # 💥 PUSHED: Specifically reward moving the hips, knees, and ankles
            reactive_reward = 3.5 * leg_movement
        else:
            # 🧍 STABLE: Penalize leg movement to force an upright stance
            reactive_reward = -2.0 * leg_movement

        # --- 4. HEAVY JITTER PENALTY ---
        # Penalize the RAW action delta so the AI's "brain" is punished for trying to twitch
        raw_action_delta = action - self.previous_raw_action
        smoothness_penalty = -250.0 * np.sum(np.square(raw_action_delta))
        
        self.previous_raw_action = np.copy(action)
        
        # Total reward calculation
        reward = 5.0 + upright_reward + reactive_reward + shoulder_penalty + smoothness_penalty
        
        # Termination (Death by falling over)
        if self.current_step > 15:
            if accel_z < 3.0:
                self.fall_counter += 1
            else:
                self.fall_counter = 0
            
            terminated = bool(self.fall_counter >= 5)
        else:
            terminated = False
            self.fall_counter = 0
            
        # Truncation (Victory by surviving 4000 steps)
        truncated = False
        if self.current_step >= 4000:
            truncated = True
            reward += 1000.0
            print(f"\n🏆 GOAL ACHIEVED! Balanced for {self.current_step} steps! 🏆\n")
            
        return state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.fall_counter = 0
        
        # Reset tracking arrays for the EMA filter
        self.previous_raw_action = np.zeros(14, dtype=np.float32)
        self.previous_smoothed_action = np.zeros(14, dtype=np.float32)
        
        # Send reset signal to Webots (fourteen 999.0s)
        reset_signal = struct.pack('14d', *([999.0] * 14))
        self.sock.send(reset_signal)
        
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        return state, {}

if __name__ == "__main__":
    env = KondoBalanceEnv()
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.02 * np.ones(n_actions))
    
    model_path = "kondo_td3_smooth.zip"
    
    if os.path.exists(model_path):
        print("Found existing smooth model! Resuming training...")
        model = TD3.load(
            "kondo_td3_smooth", 
            env=env, 
            action_noise=action_noise, 
            custom_objects={
                "learning_rate": 0.0003,
                "batch_size": 1024
            },
            tensorboard_log="./kondo_tensorboard/"
        )
    else:
        print("Initializing new TD3 Training with Smoothness Constraints...")
        model = TD3(
            "MlpPolicy", 
            env, 
            action_noise=action_noise, 
            learning_rate=0.0003,
            batch_size=1024,
            verbose=1, 
            tensorboard_log="./kondo_tensorboard/"
        )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./checkpoints/",
        name_prefix="kondo_smooth_td3"
    )
    
    try:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")
    
    model.save("kondo_td3_smooth")
    print("Smooth brain saved successfully.")