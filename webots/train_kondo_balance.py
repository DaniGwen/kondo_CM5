import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import struct
from stable_baselines3 import PPO

class KondoBalanceEnv(gym.Env):
    def __init__(self):
        super(KondoBalanceEnv, self).__init__()
        
        # Action space: Micro-adjustments for 12 leg joints
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(12,), dtype=np.float32)
        
        # Observation space: 3 Accel + 3 Gyro = 6 floats
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
        
        # Connect to Webots TCP Server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 5555))

    def step(self, action):
        # Send action to Webots (12 doubles)
        action_data = struct.pack('12d', *action)
        self.sock.send(action_data)
        
        # Receive new state from Webots (6 doubles = 48 bytes)
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        # Extract IMU data
        accel_x, accel_y, accel_z = state[0], state[1], state[2]
        
        # Reward: Maximize upright posture (Z accel near 9.81, X/Y near 0)
        upright_reward = -abs(accel_x) - abs(accel_y)
        energy_penalty = -0.1 * np.sum(np.square(action))
        reward = upright_reward + energy_penalty
        
        # Done condition: The robot fell over
        terminated = bool(accel_z < 5.0)
        truncated = False # Used if we had a maximum time limit
        
        return state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Send the secret reset signal to Webots (twelve 999.0s)
        reset_signal = struct.pack('12d', *([999.0] * 12))
        self.sock.send(reset_signal)
        
        # Read the immediate zero-state back so the TCP ping-pong stays in sync
        data = self.sock.recv(48)
        state = np.array(struct.unpack('6d', data), dtype=np.float32)
        
        return state, {}

if __name__ == "__main__":
    env = KondoBalanceEnv()
    
    print("Initializing PPO Training...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./kondo_tensorboard/")
    
    # Train the robot to balance
    model.learn(total_timesteps=100000)
    
    # Save the PyTorch model for eventual ONNX export to the CM5
    model.save("kondo_lifelike_stand")