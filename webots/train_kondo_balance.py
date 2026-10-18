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
        self.previous_action = np.zeros(14, dtype=np.float32)
        
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
        
        # Send action to Webots (14 doubles)
        action_data = struct.pack('14d', *action)
        self.sock.send(action_data)
        
        # Receive new state from Webots (6 doubles = 48 bytes)
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        # Extract IMU data (Accel X, Y, Z, Gyro Roll, Pitch, Yaw implied or standard state layout)
        accel_x, accel_y, accel_z = state[0], state[1], state[2]
        gyro_roll, gyro_pitch = state[3], state[4]
        
        upright_reward = -2.0 * (abs(accel_x) + abs(accel_y))
        energy_penalty = -0.05 * np.sum(np.square(action))
        
        # --- HEAVY ANTI-TWITCH & ADAPTIVE SMOOTHNESS PENALTY ---
        action_delta = action - self.previous_action
        
        # Measure disturbance magnitude from IMU sensors
        disturbance = abs(accel_x) + abs(accel_y) + abs(gyro_roll) + abs(gyro_pitch)
        
        # When stable (low disturbance), smoothness penalty is extremely harsh (-120.0).
        # When disturbed, penalty scales down so the robot can make fast corrections.
        dynamic_smoothness_weight = max(10.0, 120.0 - (disturbance * 40.0))
        smoothness_penalty = -dynamic_smoothness_weight * np.sum(np.square(action_delta))
        
        # Update action memory
        self.previous_action = np.copy(action)
        
        # Total reward calculation
        reward = 5.0 + upright_reward + energy_penalty + smoothness_penalty
        
        # 1. Termination (Death by falling over)
        if self.current_step > 15:
            if accel_z < 2.0:
                self.fall_counter += 1
            else:
                self.fall_counter = 0
            
            terminated = bool(self.fall_counter >= 5)
        else:
            terminated = False
            self.fall_counter = 0
            
        # 2. Truncation (Victory by surviving 1500 steps)
        truncated = False
        if self.current_step >= 1500:
            truncated = True
            reward += 1000.0
            print(f"\n🏆 GOAL ACHIEVED! Balanced for {self.current_step} steps! 🏆\n")
            
        return state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.fall_counter = 0
        self.previous_action = np.zeros(14, dtype=np.float32)
        
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
        save_freq=100000, 
        save_path="./checkpoints/",
        name_prefix="kondo_smooth_td3"
    )
    
    try:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")
    
    model.save("kondo_td3_smooth")
    print("Smooth brain saved successfully.")