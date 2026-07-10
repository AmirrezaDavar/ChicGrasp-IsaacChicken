# ChicGrasp Isaac Chicken Diffusion Policy

This repository contains the Diffusion Policy training side of the chicken grasping project.

It is based on ChicGrasp and the official Diffusion Policy codebase, with Isaac chicken-specific dataset loaders and Hydra configs.

The Isaac simulation, teleoperation collection, zarr visualization, and policy evaluation code lives in the sibling repository:

```text
/home/wanglab22/3_chicken-isaaclab
```

## Repository Roles

```text
/home/wanglab22/3_chicken-isaaclab
  Isaac simulation, assets, GELLO data collection, zarr tools, simulation evaluation.

/home/wanglab22/ChicGrasp-IsaacChicken
  Diffusion Policy algorithms, image/low-dim training configs, dataset loaders, checkpoints.
```

## Activate Training Environment

Open a new terminal and run:

```bash
source /home/wanglab22/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /home/wanglab22/ChicGrasp-IsaacChicken
unset PYTHONPATH
```

If the environment has not been created yet:

```bash
source /home/wanglab22/miniforge3/etc/profile.d/conda.sh
cd /home/wanglab22/ChicGrasp-IsaacChicken
unset PYTHONPATH

export MUJOCO_PATH=$HOME/.mujoco/mujoco-2.3.1
export MUJOCO_PLUGIN_PATH=$MUJOCO_PATH/plugin
export LD_LIBRARY_PATH=$MUJOCO_PATH/lib:$LD_LIBRARY_PATH

conda env create -f conda_environment.yaml -n robodiff
conda activate robodiff
pip install -e .
```

Check the environment:

```bash
python -c "import mujoco, dm_control, hydra, dill, torch, zarr, diffusers, robomimic; print('ok')"
```

## Expected Dataset

Collected demonstrations should be created by the Isaac repo:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/01_collect_chicken_rgb_state_demos.py \
  --out_dir ./data/chicken_rgb_state \
  --num_demos 50 \
  --save_videos
```

The training dataset path is:

```text
/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr
```

Required arrays:

```text
replay_buffer.zarr/
  data/
    camera_rgb  # (T, H, W, 3), uint8
    state       # (T, 20), float32
    action      # (T, 8), float32
  meta/
    episode_ends
```

## Train RGB + Low-Dim Diffusion Policy

This is the preferred policy because it uses both camera and robot state.

From this repo:

```bash
source /home/wanglab22/miniforge3/etc/profile.d/conda.sh
conda activate robodiff
cd /home/wanglab22/ChicGrasp-IsaacChicken
unset PYTHONPATH

python train.py \
  --config-name=train_diffusion_unet_image_isaac_chicken_workspace \
  task.dataset.zarr_path=/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  task.dataset.val_ratio=0.1 \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32 \
  dataloader.num_workers=4 \
  val_dataloader.num_workers=4 \
  training.device=cuda:0 \
  training.num_epochs=450 \
  logging.mode=offline
```

Small smoke test:

```bash
python train.py \
  --config-name=train_diffusion_unet_image_isaac_chicken_workspace \
  task.dataset.zarr_path=/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  task.dataset.val_ratio=0.25 \
  dataloader.batch_size=2 \
  val_dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  dataloader.persistent_workers=False \
  val_dataloader.persistent_workers=False \
  training.device=cuda:0 \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  logging.mode=offline
```

## Train Low-Dim Only Policy

Use this only for comparison or debugging.

```bash
python train.py \
  --config-name=train_diffusion_unet_lowdim_isaac_chicken_workspace \
  task.dataset.zarr_path=/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  +task.dataset.obs_key=state \
  task.dataset.val_ratio=0.25 \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  training.num_epochs=1000 \
  logging.mode=offline
```

## Checkpoints

Training outputs are written under:

```text
/home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/
```

Use `latest.ckpt` for evaluation unless you intentionally choose another checkpoint.

## Evaluate In Isaac Simulation

Evaluation is launched from the Isaac repo:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/03_eval_chicken_rgb_state_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

Low-dimensional checkpoint evaluation:

```bash
./isaaclab.sh -p scripts/imitation_learning/eval_chicken_diffusion_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

## Key Isaac Chicken Training Files

```text
diffusion_policy/config/train_diffusion_unet_image_isaac_chicken_workspace.yaml
diffusion_policy/config/train_diffusion_unet_lowdim_isaac_chicken_workspace.yaml
diffusion_policy/config/task/isaac_chicken_image.yaml
diffusion_policy/config/task/isaac_chicken_lowdim.yaml
diffusion_policy/dataset/isaac_chicken_image_dataset.py
diffusion_policy/dataset/isaac_chicken_dataset.py
diffusion_policy/env_runner/null_image_runner.py
diffusion_policy/env_runner/null_lowdim_runner.py
```

## Git Hygiene

Do not commit datasets or checkpoints:

```text
data/outputs/
*.ckpt
*.pt
*.pth
*.zarr/
wandb/
```

Keep algorithm/config/dataset-loader changes in this repo. Keep Isaac simulation, teleop, and evaluation changes in `/home/wanglab22/3_chicken-isaaclab`.
