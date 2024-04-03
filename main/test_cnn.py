import time
import random

from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnvCNN

NUM_EPISODES = 5000

RENDER = False
IS_SILENT = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 1
PRINT = False

BOARD_SIZE = 6

MODEL_PATH = f"trained_models_cnn_mps/{BOARD_SIZE}x{BOARD_SIZE}/ppo_snake_23000000_steps"

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

env = SnakeEnvCNN(seed=seed, board_size=BOARD_SIZE, enlarge_multiplier=84 / BOARD_SIZE, limit_step=True, is_render=RENDER, is_silent=IS_SILENT)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0
wins = 0
total_win_steps = 0

for episode in range(NUM_EPISODES):
    print(f"{episode}/{NUM_EPISODES}", end="\r")
    obs, info = env.reset()
    episode_reward = 0
    terminated = False
    
    num_steps = 0
    info = None

    sum_step_reward = 0

    if PRINT:
        print(f"=================== Episode {episode + 1} ==================")

    while not terminated:
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction
        num_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            if info["snake_size"] == env.game.grid_size:
                wins += 1
                total_win_steps += num_steps
                if PRINT:
                    print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                if PRINT:
                    print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        elif info["food_obtained"]:
            if PRINT:
                print(f"Food obtained at step {num_steps:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0 

        else:
            sum_step_reward += reward
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    if PRINT:
        snake_size = info["snake_size"]
        print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_steps}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print("Model Path:", MODEL_PATH)
print("Trials:", NUM_EPISODES)
print(f"Average Score: {total_score / NUM_EPISODES}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODES}, Win Ratio: {wins / NUM_EPISODES}, Avg Moves to Win: {'no wins' if wins == 0 else total_win_steps/wins}")
