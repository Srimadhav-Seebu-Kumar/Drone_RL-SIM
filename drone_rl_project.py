import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.grid_size = 10
        self.no_fly_zone = [(4, 4), (4, 5), (5, 4), (5, 5)]
        self.goal = (9, 9)
        self.start = (0, 0)
        self.position = list(self.start)
        self.velocity = 0.0
        self.angle = 0.0

        self.max_speed = 1.0
        self.max_turn = np.pi / 4

        self.observation_space = spaces.Box(low=np.array([0, 0, -1.0, -np.pi]), 
                                            high=np.array([self.grid_size, self.grid_size, 1.0, np.pi]), 
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-self.max_turn, -0.5]), 
                                       high=np.array([self.max_turn, 0.5]), 
                                       dtype=np.float32)

    def reset(self):
        self.position = list(self.start)
        self.velocity = 0.0
        self.angle = 0.0
        return np.array([*self.position, self.velocity, self.angle], dtype=np.float32)

    def step(self, action):
        turn, accel = action
        self.angle += np.clip(turn, -self.max_turn, self.max_turn)
        self.velocity += np.clip(accel, -0.5, 0.5)
        self.velocity = np.clip(self.velocity, 0.0, self.max_speed)

        dx = self.velocity * np.cos(self.angle)
        dy = self.velocity * np.sin(self.angle)
        self.position[0] += dx
        self.position[1] += dy

        self.position[0] = np.clip(self.position[0], 0, self.grid_size - 1)
        self.position[1] = np.clip(self.position[1], 0, self.grid_size - 1)

        pos = tuple(map(int, self.position))
        done = False
        reward = -0.1

        if pos == self.goal:
            reward = 200
            done = True
        elif pos in self.no_fly_zone:
            reward = -50
        elif self._collision():
            reward = -100
            done = True
        else:
            dist = np.linalg.norm(np.array(self.goal) - np.array(self.position))
            reward += 1 / (dist + 1e-5)

        obs = np.array([*self.position, self.velocity, self.angle], dtype=np.float32)
        return obs, reward, done, {}

    def _collision(self):
        x, y = self.position
        return not (0 <= x < self.grid_size and 0 <= y < self.grid_size)

    def render(self):
        print(f"Drone at: {self.position}, angle: {self.angle}, velocity: {self.velocity}")

def train_and_evaluate():
    env = DroneEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)

    obs = env.reset()
    trajectory = [env.position.copy()]
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append(env.position.copy())
        env.render()
        if done:
            break

    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    plt.scatter(*zip(*env.no_fly_zone), color='red', label='No-fly zone')
    plt.scatter(*env.goal, color='green', label='Goal')
    plt.title("Drone Trajectory")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, env.grid_size)
    plt.ylim(0, env.grid_size)
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
