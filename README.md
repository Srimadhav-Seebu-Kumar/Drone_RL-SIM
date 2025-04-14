# Drone RL SIM

A reinforcement learning simulation environment for autonomous drone navigation using Proximal Policy Optimization (PPO) and A* pathfinding in a 30x30 grid with dynamically generated no-fly zones.

## Features

- Custom OpenAI Gym environment for drone movement and control
- A* algorithm for optimal path planning
- PPO implementation using Stable-Baselines3
- Visualization of the drone's path, goal, and no-fly zones
- Environment constraints:
  - Turn angle limitation
  - Speed limitation
  - Collision avoidance with no-fly zones

## Files Included

| File                        | Description                                          |
|-----------------------------|------------------------------------------------------|
| `drone_rl_project.py`       | PPO-based drone navigation setup                    |
| `drone_rl_project2.py`      | Variant with minor modifications                    |
| `drone_rl_project_astar.py` | A* path integration with basic reward shaping       |
| `drone_rl_project_astar2.py`| Enhanced reward shaping and A* integration          |
| `drone_test.py`             | Test script for trained PPO model                  |
| `drone_test2.py`            | Alternate test script                               |
| `drone_ppo_model.zip`       | Trained PPO model (first version)                  |
| `drone_ppo_model_v2.zip`    | Trained PPO model (improved version)               |
| `requirements.txt`          | Required dependencies for running the code         |

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Srimadhav-Seebu-Kumar/Drone_RL-SIM.git
   cd Drone_RL-SIM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training or testing:
   ```bash
   python drone_rl_project_astar2.py
   ```

## License

This project is open source and available under the MIT License.
