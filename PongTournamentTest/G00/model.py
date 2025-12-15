from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch

class SymmetricActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_flip = False

    def extract_features(self, obs: torch.Tensor, features_extractor=None) -> torch.Tensor:
        if self.force_flip:
            obs = torch.flip(obs, dims=[-1])
        return super().extract_features(obs, features_extractor)
