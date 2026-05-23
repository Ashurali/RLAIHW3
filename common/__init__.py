"""Shared engine for the HW3 reinforcement-learning experiments.

This package holds everything that is independent of a specific task or
algorithm: configuration loading, environment factories (Atari + VizDoom),
the training loop, evaluation/recording helpers and plotting. The thin
entry-point scripts under ``pong/`` and ``vizdoom/`` only pick an SB3 algorithm
class and a config file, then defer to :func:`common.train_core.run_training`.
"""
