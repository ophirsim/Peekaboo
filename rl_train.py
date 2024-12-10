import robosuite as suite
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
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
#need to use the gymwrapper to be compatible with stable_baselines3
#they have some check_env function to ensure that your environment is valid and compatible

block_lifting_env = Monitor(block_lifting_env)
block_lifting_env = DummyVecEnv([lambda: block_lifting_env])

policy_kwargs = dict(
    features_extractor_class=DINOv2FeatureExtractor,
    features_extractor_kwargs=dict(embed_dim=384),
    net_arch=[256, 256] #might need to make bigger bc there are 384 features from dinov2 small
)

'''
#need to figure out how exactly to set these training parameters
model = SAC(
    'MlpPolicy',                   # Policy type
    block_lifting_env,             # Environment
    policy_kwargs=policy_kwargs,   # Policy network arguments
    verbose=1,                     # Verbosity level
    gamma=0.99,                    # Discount factor
    learning_rate=3e-4,            # Learning rate for both policy and Q networks
    buffer_size=100000,           # Replay buffer size
    batch_size=256,                # Batch size for training
    ent_coef='auto',               # Entropy coefficient (auto means it will be tuned)
    tau=0.005,                     # Soft target tau for target updates
    target_update_interval=1,      # Number of steps between target updates
    learning_starts=1000,          # Number of steps before training starts
    gradient_steps=100,           # Number of gradient steps per training step
    train_freq=(1000, 'step'),     # Training frequency
    tensorboard_log="./sac_tensorboard/",
)
'''

model = PPO(
    'MlpPolicy',                   # Policy type
    block_lifting_env,             # Your custom environment
    policy_kwargs=policy_kwargs,   # Policy arguments with feature extractor
    verbose=1,                     # Verbosity level
    learning_rate=3e-4,            # Learning rate for policy and value networks
    n_steps=1024,                  # Number of steps per update (increase if episodes are long)
    batch_size=256,                # Batch size for training
    gamma=0.99,                    # Discount factor
    gae_lambda=0.95,               # Generalized Advantage Estimation parameter
    clip_range=0.2,                # PPO clipping parameter
    #ent_coef=0.01,                 # Entropy coefficient for exploration
    vf_coef=0.5,                   # Value function coefficient in loss
    max_grad_norm=0.5,             # Gradient clipping
    n_epochs=10,                   # Number of epochs per update
    stats_window_size=2,           
    tensorboard_log="./ppo_tensorboard/",  # TensorBoard log directory
)

model_path = "./full_randomization_training/checkpoints/ppo_model_180000_steps"
model = PPO.load(model_path, env=block_lifting_env)

# Save a checkpoint every 100,000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,      # Save every 100,000 timesteps
    save_path='./checkpoints/',  # Directory to save checkpoints
    name_prefix='ppo_model'  # File name prefix
)

model.learn(total_timesteps=int(820000), progress_bar = True, callback=checkpoint_callback)

model_path = "./ppo_model/model"
model.save(model_path)