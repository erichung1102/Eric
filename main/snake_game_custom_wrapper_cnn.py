import math

import gymnasium
import numpy as np

from snake_game import SnakeGame

class SnakeEnvCNN(gymnasium.Env):
    def __init__(self, seed=0, board_size=12, reward_scale=0.1, enlarge_multiplier=7, is_render=False, is_silent=True, limit_step=True, cell_size=40, border_size=20):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, is_render=is_render, is_silent=is_silent, cell_size=cell_size, border_size=border_size)
        self.game.reset()

        self.seed = seed
        self.reward_scale = reward_scale

        self.is_silent = is_silent

        self.action_space = gymnasium.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.enlarge_multiplier = enlarge_multiplier
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.terminated = False

        if limit_step:
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    def reset(self, seed=None, **kwargs):
        self.game.reset(seed=seed)
        super().reset(seed=seed)

        self.terminated = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs, {}
    
    def step(self, action):
        self.terminated, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if info["snake_size"] == self.grid_size: # Snake fills up the entire board, game over.
            reward = self.max_growth # Victory reward
            self.terminated = True
        
        elif self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.terminated = True
        
        elif self.terminated: # Snake bumps into wall or itself, game over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)  
            # Game Over penalty is based on board size. 
            reward = -self.max_growth
          
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0 # Reset reward step counter
        
        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
        
        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return obs, reward * self.reward_scale, self.terminated, False, info
    
    def render(self, x_offset=0):
        self.game.render(x_offset=x_offset)

    def close(self):
        self.game.close()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be popped in the current step.
        game_over = (
            (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
            or row < 0
            or row >= self.board_size
            or col < 0
            or col >= self.board_size
        )
            
        if game_over:
            return False
        return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; SnakeTAIL: Blue; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs,) * 3, axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [0, 0, 255]

        # Set the food to red
        obs[self.game.food] = [255, 0, 0]

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, self.enlarge_multiplier, axis=0), self.enlarge_multiplier, axis=1)

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(is_render=True, is_silent=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     terminated = False
    #     i = 0
    #     while not terminated:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, terminated, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))

from gymnasium.utils.env_checker import check_env

if __name__ == '__main__':
    env = SnakeEnvCNN()
    env.reset()
    check_env(env)
    print("observation shape:", env._generate_observation().shape)