import os
from typing import Any, Dict, Optional, Tuple

# import cv2
import gym
import numpy as np
import torch as th
from pathlib import Path

from dalle_pytorch import DiscreteVAE
from dalle_pytorch.data_loader import preprocess_image


class AutoencoderWrapper(gym.Wrapper):
    """
    Gym wrapper to encode image and reduce input dimension
    using pre-trained auto-encoder
    (only the encoder part is used here, decoder part can be used for debug)

    :param env: Gym environment
    :param ae_path: Path to the autoencoder
    """

    def __init__(self, env: gym.Env, ae_path: Optional[str] = os.environ.get("AAE_PATH")):  # noqa: B008
        super().__init__(env)

        ae_path = Path(ae_path)
        assert ae_path.exists(), 'AE model file does not exist'
        assert not ae_path.is_dir(), \
            ('Cannot load VAE model from directory; please use a '
             'standard *.pt checkpoint. ')

        loaded_obj = th.load(str(ae_path))
        ae_params, weights = loaded_obj['hparams'], loaded_obj['weights']
        self.ae = DiscreteVAE(**ae_params).cuda()
        self.ae.load_state_dict(weights)

        # Update observation space
        obs_shape_dim = (self.ae.image_size[0] // (2 ** self.ae.num_layers)) * (self.ae.image_size[1] // (2 ** self.ae.num_layers))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape_dim + 1,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        # Important: Convert to BGR to match OpenCV convention
        obs = self.env.reset()
        obs = preprocess_image(obs, convert_to_rgb=True)
        encoded_image = self.ae.get_codebook_indices(th.as_tensor(obs).unsqueeze(0).cuda())
        new_obs = np.concatenate([encoded_image.cpu().numpy().flatten(), [0.0]])
        return new_obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        # encoded_img = self.ae.encode_from_raw_image(obs[:, :, ::-1])
        # reconstructed_img = self.ae.decode(encoded_img)[0]
        # cv2.imshow("Original", obs[:, :, ::-1])
        # cv2.imshow("Reconstruction", reconstructed_img)
        # cv2.waitKey(0)
        obs = preprocess_image(obs, convert_to_rgb=True)
        encoded_image = self.ae.get_codebook_indices(th.as_tensor(obs).unsqueeze(0).cuda())
        speed = infos["speed"]
        new_obs = np.concatenate([encoded_image.cpu().numpy().flatten(), [speed]])
        return new_obs, reward, done, infos
