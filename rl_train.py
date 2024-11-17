import robosuite as suite
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import SAC
from robosuite.wrappers.gym_wrapper import GymWrapper
from vision_encoder import DINOv2FeatureExtractor


config = {
    "horizon": 500,
    "control_freq": 20,
    "reward_shaping": True,
    "reward_scale": 1.0,
    "use_camera_obs": True,
    "ignore_done": True,
    "hard_reset": False,
}

# this should be used during training to speed up training
# A renderer should be used if you're visualizing rollouts!
config["has_renderer"] = False
config["has_offscreen_renderer"] = True

# Block Lifting
env = suite.make(
    env_name="Lift", 
    robots="Panda",
    **config,
)

block_lifting_env = GymWrapper(env)
#need to use the gymwrapper to be compatible with stable_baselines3
#they have some check_env function to ensure that your environment is valid and compatible

policy_kwargs = dict(
    features_extractor_class=DINOv2FeatureExtractor,
    features_extractor_kwargs=dict(embed_dim=768),
    net_arch=[256, 256] #might need to make bigger bc there are 768 features from dinov2
)

#need to figure out how exactly to set these training parameters
model = SAC(
    'MlpPolicy',                   # Policy type
    block_lifting_env,             # Environment
    policy_kwargs=policy_kwargs,   # Policy network arguments
    verbose=1,                     # Verbosity level
    gamma=0.99,                    # Discount factor
    learning_rate=3e-4,            # Learning rate for both policy and Q networks
    buffer_size=1000000,           # Replay buffer size
    batch_size=256,                # Batch size for training
    ent_coef='auto',               # Entropy coefficient (auto means it will be tuned)
    tau=0.005,                     # Soft target tau for target updates
    target_update_interval=1,      # Number of steps between target updates
    learning_starts=1000,          # Number of steps before training starts
    gradient_steps=1000,           # Number of gradient steps per training step
    train_freq=(5000, 'step'),     # Training frequency
)

'''
Using cuda device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.

this is the output that i get, might need to actually use a Monitor and VecEnv wrapper in the future
'''


model.learn(total_timesteps=int(1e7), progress_bar = True)

model_path = "./sac_model/model"
model.save(model_path)