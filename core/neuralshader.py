from core.fc import FC
from core.gfft import GaussianFourierFeatureTransform

import numpy as np
import torch

class NeuralShader(torch.nn.Module):

    def __init__(self,
                 input_size = 27,
                 hidden_features_size=256,
                 hidden_features_layers=7,
                 activation='relu',
                 last_activation=None,
                 mapping_size=256,
                 device='cpu'):

        super().__init__()
        self.diffuse = FC(input_size, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None).to(device)
        self.specular = FC(hidden_features_size, 3, [hidden_features_size // 2], activation, last_activation).to(device)

        # Store the config
        self._config = {
            'hidden_features_size': hidden_features_size,
            'hidden_features_layers': hidden_features_layers,
            'activation': activation,
            'last_activation': last_activation,
            'mapping_size': mapping_size,
        }

    def forward(self, feat):
        h = self.diffuse(feat)
        return self.specular(h)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device)

        # Convert data between versions
        version = data['version']

        shader = cls(**data['config'], device=device)
        shader.load_state_dict(data['state_dict'])

        if version < 2 and isinstance(shader.fourier_feature_transform, GaussianFourierFeatureTransform):
            print("Warning: B matrix for GFFT features is not stored in checkpoints of versions < 2")
        elif isinstance(shader.fourier_feature_transform, GaussianFourierFeatureTransform):
            shader.fourier_feature_transform.B = data['B']
        return shader

    def save(self, path):
        data = {
            'version': 2,
            'config': self._config,
            'state_dict': self.state_dict()
        }

        if isinstance(self.fourier_feature_transform, GaussianFourierFeatureTransform):
            data['B'] = self.fourier_feature_transform.B
        torch.save(data, path)