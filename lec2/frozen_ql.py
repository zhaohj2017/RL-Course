#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified by zhaohj Nov-16-2020

# -*- coding: utf-8 -*-

import gym
from gridworld import FrozenLakeWapper
import sarsa
import time

# episode training
def train_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
                        # obs: initial state
    action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # update current state and action

        total_reward += reward
        total_steps += 1  # 计算step数

        if render:
            env.render()  #渲染新的一帧图形
                            # plot for each learning step in this episode
        if done: 
            break # reach terminal state, end the learning of this episode

    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy, no epsilon randomness
        next_obs, reward, done, _ = env.step(action)

        # update state and reward
        obs = next_obs
        total_reward += reward

        time.sleep(0.5) # for visualization
        env.render()

        # reach terminal state
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # setup env
    env = gym.make("FrozenLake-v0", is_slippery=True)  # 0 left, 1 down, 2 right, 3 up
                                                        # probabilistic transitions
    env = FrozenLakeWapper(env)

    # generate agent
    agent = sarsa.SarsaAgent(
        obs_n=env.observation_space.n, # the dimension of the state space of env
        act_n=env.action_space.n, # the dimension of the action space of env
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    is_render = False
    for episode in range(1000): # learn 1000 episode
        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False

        ep_reward, ep_steps = train_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))


    print("\nPlease input Enter Key to test the learned policy: "), input()
    # 训练结束，查看算法效果
    test_episode(env, agent)


if __name__ == "__main__":
    main()