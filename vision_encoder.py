import timm
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

class DINOv2FeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using the pretrained DINOv2 base model from timm.
    This version includes preprocessing as part of the feature extractor.
    """
    def __init__(self, observation_space, model_name= "hf_hub:timm/vit_small_patch14_dinov2.lvd142m", embed_dim=384):
        super().__init__(observation_space, features_dim=embed_dim)
        
        # Load pretrained DINOv2 model
        self.dino_model = timm.create_model("hf_hub:timm/vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        # Freeze the model parameters
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # Get preprocessing configuration
        config = timm.data.resolve_data_config({}, model=self.dino_model)
        self.preprocess = timm.data.create_transform(**config)

    def forward(self, observations):
        """
        Forward pass through the DINOv2 model.
        Assumes observations are raw images (batch of N, H, W, C).
        """
        obs = observations.squeeze(0)[:196608]
        obs = obs.reshape(256, 256, 3)
        obs = obs.unsqueeze(0)
        
        # Convert (N, H, W, C) to (N, C, H, W) for PyTorch
        obs = obs.permute(0, 3, 1, 2)

        # Apply preprocessing
        obs = torch.stack([self.preprocess(img) for img in obs])

        # Extract features using the DINOv2 model
        return self.dino_model(obs)
    


    