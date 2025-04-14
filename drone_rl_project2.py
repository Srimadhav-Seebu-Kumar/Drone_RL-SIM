# drone_rl_dynamic.py

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gym import spaces
import random

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.grid_size = 10
        self.max_speed = 1.0
        self.max_turn = np.pi / 6  # angle
        self.n_no_fly_zones = 6
        self.no_fly_zone_size = 1.0
        self.action_space = spaces.Box(low=np.array([-self.max_turn, 0.0]), high=np.array([self.max_turn, self.max_speed]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(4,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.position = np.array([0.0, 0.0])
        self.angle = 0.0
        self.velocity = 0.0
        self.goal = np.array([9.0, 9.0])
        self.no_fly_zones = self.generate_no_fly_zones()
        return self._get_obs()

    def generate_no_fly_zones(self):
        zones = []
        while len(zones) < self.n_no_fly_zones:
            x = random.uniform(0, self.grid_size - self.no_fly_zone_size)
            y = random.uniform(0, self.grid_size - self.no_fly_zone_size)
            zone = ((x, x + self.no_fly_zone_size), (y, y + self.no_fly_zone_size))
            if not self._in_zone(self.position, zone) and not self._in_zone(self.goal, zone):
                zones.append(zone)
        return zones

    def step(self, action):
        turn, throttle = action
        self.angle += np.clip(turn, -self.max_turn, self.max_turn)
        self.velocity = np.clip(throttle, 0.0, self.max_speed)
        dx = self.velocity * np.cos(self.angle)
        dy = self.velocity * np.sin(self.angle)
        new_position = self.position + np.array([dx, dy])

        if self._in_bounds(new_position) and not self._in_no_fly_zone(new_position):
            self.position = new_position
        else:
            self.velocity = 0.0

        done = np.linalg.norm(self.position - self.goal) < 0.5
        reward = 100.0 if done else -0.1

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([*self.position, self.angle, self.velocity], dtype=np.float32)

    def _in_bounds(self, pos):
        return 0 <= pos[0] <= self.grid_size and 0 <= pos[1] <= self.grid_size

    def _in_no_fly_zone(self, pos):
        return any(self._in_zone(pos, zone) for zone in self.no_fly_zones)

    def _in_zone(self, pos, zone):
        (x_min, x_max), (y_min, y_max) = zone
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max

    def render(self, mode="human"):
        plt.clf()
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.plot(self.position[0], self.position[1], "bo", label="Drone")
        plt.plot(self.goal[0], self.goal[1], "go", label="Goal")
        for zone in self.no_fly_zones:
            (x_min, x_max), (y_min, y_max) = zone
            plt.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], color='red', alpha=0.4)
        plt.pause(0.01)

def train_and_evaluate():
    env = DroneEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f"Drone at: {env.position}, angle: {env.angle}, velocity: {env.velocity}")
        env.render()

if __name__ == "__main__":
    train_and_evaluate()
