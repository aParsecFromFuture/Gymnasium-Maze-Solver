import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class MouseCatEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, render_mode=None, size=5, n_obstacles=10):
        super().__init__()

        self.size = size
        self.window_size = 512
        self.n_obstacles = n_obstacles

        # Images for drawing

        pix_square_size = self.window_size / self.size

        self._mouse_img = pygame.image.load('maze_worlds/resources/mouse.jpg')
        self._mouse_img = pygame.transform.scale(self._mouse_img, (pix_square_size, pix_square_size))

        self._cat_img = pygame.image.load('maze_worlds/resources/cat.jpg')
        self._cat_img = pygame.transform.scale(self._cat_img, (pix_square_size, pix_square_size))

        self._cheese_img = pygame.image.load('maze_worlds/resources/cheese.jpg')
        self._cheese_img = pygame.transform.scale(self._cheese_img, (pix_square_size, pix_square_size))

        # State space, action space and transition rules

        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        self.action_space = spaces.Discrete(4)
        self._transition = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            'distance': np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.observation_space.sample()

        while True:
            self._target_location = self.observation_space.sample()
            if np.array_equal(self._target_location, self._agent_location):
                continue
            break

        while True:
            self._enemy_location = self.observation_space.sample()
            if np.array_equal(self._enemy_location, self._agent_location):
                continue
            if np.array_equal(self._enemy_location, self._agent_location):
                continue
            break

        self._obstacle_locations = []
        for _ in range(self.n_obstacles):
            while True:
                obstacle = self.observation_space.sample()
                if np.array_equal(obstacle, self._agent_location):
                    continue
                if np.array_equal(obstacle, self._target_location):
                    continue
                if np.array_equal(obstacle, self._enemy_location):
                    continue
                if any([np.array_equal(obstacle, x) for x in self._obstacle_locations]):
                    continue
                self._obstacle_locations.append(obstacle)
                break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._transition[action]

        if not any([np.array_equal(self._agent_location + direction, wall) for wall in self._obstacle_locations]):
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        does_win = np.array_equal(self._agent_location, self._target_location)
        does_lose = np.array_equal(self._agent_location, self._enemy_location)
        terminated = does_win or does_lose

        reward = 10 if does_win else -10 if does_lose else -0.05
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        self.render_mode = 'human'
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        canvas.blit(
            self._cheese_img,
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        canvas.blit(
            self._mouse_img,
            pygame.Rect(
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the enemy
        canvas.blit(
            self._cat_img,
            pygame.Rect(
                pix_square_size * self._enemy_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the walls
        for obstacle in self._obstacle_locations:
            pygame.draw.rect(
                canvas,
                (50, 50, 50),
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

