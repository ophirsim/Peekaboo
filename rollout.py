import robosuite as suite
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import SAC, PPO
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
import matplotlib.pyplot as plt
from vision_encoder import DINOv2FeatureExtractor
from randomized_env import CustomLiftWithWall

config = {
    "horizon": 500,
    "control_freq": 20,
    "reward_shaping": True,
    "reward_scale": 1.0,
    "camera_names": "robot0_eye_in_hand",
    "camera_heights": 224, 
    "camera_widths": 224,
    "use_camera_obs": True,
    "use_object_obs": False,
    "ignore_done": True,
    "hard_reset": False,
}

# this should be used during training to speed up training
# A renderer should be used if you're visualizing rollouts!
config["has_renderer"] = False
config["has_offscreen_renderer"] = True

randomize_arm = True # flag to randomize arm
placement_initializer = UniformRandomSampler(
    name="ObjectSampler",
    x_range=[-0.3, 0.3],
    y_range=[-0.3, 0.3],
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=np.array((0, 0, 0.8)),
    z_offset=0.01,
)

# Block Lifting
env = suite.make(
    env_name="CustomLiftWithWall", 
    robots="Panda",
    initialization_noise={'magnitude': 0.3 if randomize_arm else 0.0, 'type': 'uniform'},
    placement_initializer=placement_initializer,
    **config,
)

block_lifting_env = GymWrapper(env)

#model_path = "./ppo_model/model"
#model = PPO.load(model_path, env=block_lifting_env)

policy_kwargs = dict(
    features_extractor_class=DINOv2FeatureExtractor,
    features_extractor_kwargs=dict(embed_dim=384),
    net_arch=[256, 256] #might need to make bigger bc there are 384 features from dinov2 small
)

model = PPO(
    'MlpPolicy',                   # Policy type
    block_lifting_env,             # Your custom environment
    policy_kwargs=policy_kwargs,   # Policy arguments with feature extractor
    verbose=1,                     # Verbosity level
    learning_rate=3e-4,            # Learning rate for policy and value networks
    n_steps=2048,                  # Number of steps per update (increase if episodes are long)
    batch_size=256,                # Batch size for training
    gamma=0.99,                    # Discount factor
    gae_lambda=0.95,               # Generalized Advantage Estimation parameter
    clip_range=0.2,                # PPO clipping parameter
    ent_coef=0.01,                 # Entropy coefficient for exploration
    vf_coef=0.5,                   # Value function coefficient in loss
    max_grad_norm=0.5,             # Gradient clipping
    n_epochs=10,                   # Number of epochs per update
    stats_window_size=1,           
    tensorboard_log="./ppo_tensorboard/",  # TensorBoard log directory
)

num_episodes = 10
dones = []
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"Starting Episode {episode + 1}")
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        dones.append(done)

        if step >= config["horizon"]:
            break

    print(f"Episode {episode + 1} ended with reward: {total_reward}")

print(dones)

env.close()
