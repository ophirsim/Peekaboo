import robosuite as suite
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import SAC
from robosuite.wrappers.gym_wrapper import GymWrapper
import matplotlib.pyplot as plt

config = {
    "horizon": 500,
    "control_freq": 20,
    "reward_shaping": True,
    "reward_scale": 1.0,
    "use_camera_obs": True,
    "ignore_done": False,
    "hard_reset": False,
}

# this should be used during training to speed up training
# A renderer should be used if you're visualizing rollouts!
config["has_renderer"] = True
config["has_offscreen_renderer"] = False

# Block Lifting
env = suite.make(
    env_name="Lift", 
    robots="Panda",
    **config,
)

block_lifting_env = GymWrapper(env)

model_path = "./sac_model/model"
model = SAC.load(model_path, env=block_lifting_env)

def visualize_rollout(env, model, num_episodes=1):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"Starting Episode {episode + 1}")
        while not done:
            env.render()

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if step >= config["horizon"]:
                break

        print(f"Episode {episode + 1} ended with reward: {total_reward}")

    env.close()

visualize_rollout(block_lifting_env, model, num_episodes=1)
