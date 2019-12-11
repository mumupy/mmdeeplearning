#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 20:48
# @Author  : ganliang
# @File    : tf_gym.py
# @Desc    : 强化学习
import gym
import atari_py
from src.config import logger


def gym_basic():
    env_name = "Breakout-v0"
    # env_name = "CartPole-v0"
    env = gym.make(env_name)
    obs = env.reset()
    logger.info(obs.shape)
