import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gym import spaces
import random
import heapq  

def astar_path(start, goal, grid_size, no_fly_zones):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def neighbors(pos):
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        result = []
        for dx, dy in directions:
            neighbor = (pos[0] + dx, pos[1] + dy)
            if (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                if not any(zone[0][0] <= neighbor[0] <= zone[0][1] and zone[1][0] <= neighbor[1] <= zone[1][1] for zone in no_fly_zones):
                    result.append(neighbor)
        return result

    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for next in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    current = goal
    path = []
    while current and current in came_from:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.grid_size = 10
        self.max_speed = 1.0
        self.max_turn = np.pi / 8  # angle
        self.n_no_fly_zones = 20
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
        self.astar_path = astar_path(self.position, self.goal, self.grid_size, self.no_fly_zones)
        self.path_index = 0
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
        prev_distance_to_goal = np.linalg.norm(self.position - self.goal)

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
            return self._get_obs(), -2.0, False, {} 

        distance_to_goal = np.linalg.norm(self.position - self.goal)
        done = distance_to_goal < 1.0

        reward = 0.0

        if done:
            reward += 200.0 
        else:
            progress_reward = prev_distance_to_goal - distance_to_goal
            reward += 3.0 * progress_reward 

            best_dist = float('inf')
            closest_index = self.path_index
            for i in range(self.path_index, min(self.path_index + 8, len(self.astar_path))):
                path_point = np.array(self.astar_path[i]) + 0.5
                dist = np.linalg.norm(self.position - path_point)
                if dist < best_dist:
                    best_dist = dist
                    closest_index = i

            reward += max(0, 6.0 - 1.2 * best_dist)

            if closest_index < self.path_index:
                reward -= 2.0  
            elif best_dist > 1.5:
                reward -= 1.0 

            if best_dist < 0.75:
                self.path_index = max(self.path_index, closest_index + 1)

            reward -= 0.02 * abs(turn)
            if self.velocity < 0.1:
                reward -= 0.05

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
        
# A* path
        for i in range(len(self.astar_path)-1):
            plt.plot([self.astar_path[i][0], self.astar_path[i+1][0]], 
                     [self.astar_path[i][1], self.astar_path[i+1][1]], 'r-', lw=2, alpha=0.7)

# No-fly zones
        for zone in self.no_fly_zones:
            (x_min, x_max), (y_min, y_max) = zone
            plt.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], color='red', alpha=0.4)
        
        plt.pause(0.01)

def train_and_evaluate():
    env = DroneEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)  

    model.save("drone_ppo_model_1")

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f"Drone at: {env.position}, angle: {env.angle}, velocity: {env.velocity}")
        env.render()

if __name__ == "__main__":
    train_and_evaluate()

