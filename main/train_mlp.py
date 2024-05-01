import os
import sys
import random

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_mlp import SnakeEnvMLP

BOARD_SIZE = 4

NUM_ENV = 32
LOG_DIR = "logs"

SAVE_NAME = f"{BOARD_SIZE}x{BOARD_SIZE}_2"

# Set the save directory
SAVE_DIR = f"trained_models_mlp/" + SAVE_NAME

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(board_size, seed=0):
    def _init():
        env = SnakeEnvMLP(seed=seed, board_size=board_size)
        env = ActionMasker(env, SnakeEnvMLP.get_action_mask)
        env = Monitor(env)
        return env
    return _init

def main(board_size, log_dir, save_dir, save_name, num_env):
    os.makedirs(log_dir, exist_ok=True)

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < num_env:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(board_size, seed=s) for s in seed_set])

    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # # Instantiate a PPO agent
    model = MaskablePPO(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=log_dir
    )
    
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback],
            tb_log_name=f"ppo_mlp" + save_name
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_final.zip"))

if __name__ == "__main__":
    main(BOARD_SIZE, SAVE_DIR, SAVE_NAME, NUM_ENV)
