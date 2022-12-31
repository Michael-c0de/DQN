"""渲染CartPole环境, 并采用学习到的策略和环境交互"""

import gym
import torch
from util import DQN
from train import hidden_dim, lr, gamma, epsilon, target_update, device


def load_agent():
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr,
                gamma, epsilon, target_update, device)
    agent.q_net.load_state_dict(torch.load("./qnet.pkl"))
    return agent, env


if __name__ == '__main__':

    agent, env = load_agent()
    state = env.reset()

    score = 0

    done = False
    while done == False:
        env.render()

        action=agent.take_action(state)
        print("%d" % action)
        state, reward, done, _ = env.step(action)
        
        score += reward
    env.close()

    print("score - %d" % score)