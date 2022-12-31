"""DQN模型训练"""

import random
import torch
import gym
import numpy as np
import seaborn
import matplotlib.pyplot as plt 
from util import *

def trainDQN(num_episodes, env:gym.Env, agent:DQN, replay_buffer:ReplayBuffer):
    """
    Parameters:

    num_episodes - 训练轮数

    env - gym.Env

    agent - 智能体

    replay_buffer - 回放池

    """
    return_list = []
    minimal_size = 500
    for i_episode in range(num_episodes):
        episode_return = 0
        state=env.reset()
        done=False
        while not done:
            action=agent.take_action(state)
            next_state, reward,done,_=env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state=next_state
            episode_return+=reward

            if replay_buffer.size()>minimal_size:
                b_s, b_a, b_r, b_ns, b_d=replay_buffer.sample(batch_size)
                transition_dict={
                    "states":b_s,
                    "actions":b_a,
                    "next_states":b_ns,
                    "rewards":b_r,
                    "dones":b_d
                }
                agent.update(transition_dict)
        return_list.append(episode_return)
        
        if(i_episode+1)%10==0:
            print("episode %3d \t  mean return - %.1f" % (i_episode, sum(return_list[-10:])/10))
    return return_list

lr = 5e-3
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000

batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,target_update, device)

if __name__=='__main__':
    return_list=trainDQN(num_episodes,env,agent,replay_buffer)
    # 保存returnlist曲线
    fig=seaborn.lineplot(data=return_list)
    fig.get_figure().savefig("./return.jpg",dpi=400)
    
    # 保存网络参数
    torch.save(agent.q_net.state_dict(),"./qnet.pkl")