"""Dataset for Isaac Sim chicken-lift demonstrations (Zarr format).

The Zarr replay buffer is created by
  3_chicken-isaaclab/scripts/imitation_learning/collect_isaac_demos.py

Zarr layout:
  data/
    obs     : (T, 13)  float32  — [ee_pos(3), ee_euler(3), ck_pos(3), ck_euler(3), gripper(1)]
    action  : (T, 7)   float32  — [ee_delta_pos(3), ee_delta_euler(3), gripper_binary(1)]
  meta/
    episode_ends : (N,) int64
"""

from typing import Dict, Optional
import copy
import torch
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply


class IsaacChickenDataset(BaseLowdimDataset):
    """Loads Isaac Sim chicken-lift demos from a Zarr replay buffer."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        obs_key: str = "obs",
        action_key: str = "action",
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key]
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.obs_key    = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon    = horizon
        self.pad_before = pad_before
        self.pad_after  = pad_after

    # ── normalizer ────────────────────────────────────────────────────────────

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "obs":    self.replay_buffer[self.obs_key],
            "action": self.replay_buffer[self.action_key],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    # ── validation split ─────────────────────────────────────────────────────

    def get_validation_dataset(self) -> "IsaacChickenDataset":
        val = copy.copy(self)
        val.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val.train_mask = ~self.train_mask
        return val

    # ── PyTorch interface ─────────────────────────────────────────────────────

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = {
            "obs":    sample[self.obs_key].astype(np.float32),
            "action": sample[self.action_key].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)
