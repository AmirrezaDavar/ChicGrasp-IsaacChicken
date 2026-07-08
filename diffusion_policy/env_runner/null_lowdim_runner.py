"""No-op runner for offline-only training (no online env evaluation)."""

from typing import Dict
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class NullLowdimRunner(BaseLowdimRunner):
    """Returns an empty log dict — used when online rollout evaluation is not needed."""

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return {}
