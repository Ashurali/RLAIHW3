"""VizDoom -> SB3 image pipeline.

The Farama Gymnasium wrapper for ViZDoom returns a *Dict* observation: a
``screen`` image plus a ``gamevariables`` vector. SB3's ``CnnPolicy`` expects a
single image tensor, so this wrapper drops the game variables and converts the
screen to the same 84x84 (optionally grayscale) uint8 format used for Atari.

With a subsequent ``VecFrameStack(4)`` this reproduces the classic Atari DQN/PPO
front-end, giving Pong and VizDoom an identical observation pipeline so the
value-based vs policy-based comparison is not confounded by preprocessing.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Importing the wrapper module registers all "Vizdoom*-v*" Gymnasium ids as a
# side effect. Requires vizdoom >= 1.2 (Gymnasium support).
import vizdoom.gymnasium_wrapper  # noqa: F401


class VizDoomScreenWrapper(gym.ObservationWrapper):
    """Extract the screen buffer, resize it and optionally grayscale it.

    Output observation: a uint8 array of shape ``(H, W, C)`` (channel-last,
    ``C == 1`` when grayscale else ``3``). Channel-last is what SB3's
    ``VecTransposeImage`` expects before it auto-converts to channel-first.
    """

    def __init__(self, env, shape=(84, 84), grayscale: bool = True):
        super().__init__(env)
        self.shape = tuple(shape)
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.shape[0], self.shape[1], channels),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # The Dict obs carries the frame under "screen"; tolerate a bare array.
        screen = obs["screen"] if isinstance(obs, dict) else obs
        screen = np.asarray(screen)

        # Normalize to channel-last (H, W, C); some screen formats are (C, H, W).
        if screen.ndim == 3 and screen.shape[0] in (1, 3) and screen.shape[2] not in (1, 3):
            screen = np.transpose(screen, (1, 2, 0))

        if self.grayscale:
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # cv2.resize takes (width, height); self.shape is (height, width).
        screen = cv2.resize(
            screen, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA
        )

        if self.grayscale:
            screen = screen[:, :, None]
        return screen.astype(np.uint8)
