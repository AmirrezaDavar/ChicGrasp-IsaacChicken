"""Isaac chicken RGB+state zarr dataset for Diffusion Policy."""

from __future__ import annotations

from typing import Dict, Optional
import copy

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class IsaacChickenImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        n_obs_steps: Optional[int] = 2,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        image_key: str = "camera_rgb",
        state_key: str = "state",
        action_key: str = "action",
    ):
        super().__init__()

        self.image_key = image_key
        self.state_key = state_key
        self.action_key = action_key
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[image_key, state_key, action_key],
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)

        key_first_k = {}
        if n_obs_steps is not None:
            key_first_k[image_key] = n_obs_steps
            key_first_k[state_key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.train_mask = train_mask

    def get_validation_dataset(self) -> "IsaacChickenImageDataset":
        val = copy.copy(self)
        val.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k={
                self.image_key: self.n_obs_steps,
                self.state_key: self.n_obs_steps,
            },
        )
        val.train_mask = ~self.train_mask
        return val

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.action_key]
        )
        normalizer[self.state_key] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.state_key]
        )
        normalizer[self.image_key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        obs_slice = slice(self.n_obs_steps)

        image = sample[self.image_key][obs_slice]
        if image.dtype != np.uint8:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0

        data = {
            "obs": {
                self.image_key: image,
                self.state_key: sample[self.state_key][obs_slice].astype(np.float32),
            },
            "action": sample[self.action_key].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)
