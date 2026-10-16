import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import struct
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class KondoBalanceEnv(gym.Env):
    def __init__(self):
        super(KondoBalanceEnv, self).__init__()
        
       # 12 leg joints limited to +/- 0.1. 2 shoulder joints given +/- 0.5 to swing.
        low_bounds = np.array([-0.1]*12 + [-0.2, -0.2, -0.2, -0.2], dtype=np.float32)
        high_bounds = np.array([ 0.1]*12 + [ 0.2,  0.2,  0.2,  0.2], dtype=np.float32)
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(16,), dtype=np.float32)
        
        # Observation space: 3 Accel + 3 Gyro = 6 floats
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
        
        # Connect to Webots TCP Server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 5555))

    def step(self, action):
        # Send action to Webots (16 doubles)
        action_data = struct.pack('16d', *action)
        self.sock.send(action_data)
        
        # Receive new state from Webots (6 doubles = 48 bytes)
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        # Extract IMU data
        accel_x, accel_y, accel_z = state[0], state[1], state[2]
        
        upright_reward = -2.0 * (abs(accel_x) + abs(accel_y))
        
        energy_penalty = -0.1 * np.sum(np.square(action))
        flat_foot_penalty = -20.0 * (abs(action[5]) + abs(action[11]))
        
        reward = 5.0 + upright_reward + energy_penalty + flat_foot_penalty
        
        # Revert to 5.0 so the AI is allowed to experience extreme recovery angles
        terminated = bool(accel_z < 7.0)
        truncated = False # Used if we had a maximum time limit
        
        return state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Send the secret reset signal to Webots (sixteen 999.0s)
        reset_signal = struct.pack('16d', *([999.0] * 16))
        self.sock.send(reset_signal)
        
        # Read the immediate zero-state back so the TCP ping-pong stays in sync
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        return state, {}

if __name__ == "__main__":
    env = KondoBalanceEnv()
    
    print("Initializing PPO Training...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./kondo_tensorboard/")
    
    # Save a backup model every 100,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path="./checkpoints/",
        name_prefix="kondo_stand"
    )
    
    # Train for 2 Million steps (approx. 1.5 to 2 hours)
    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    
    model.save("kondo_lifelike_stand_final")