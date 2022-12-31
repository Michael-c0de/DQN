import random
import numpy as np
import collections
import torch
import torch.nn.functional as F

class ReplayBuffer:
    """
    经验回放池
    
    """
    def __init__(self, capacity):
        """

        Parameters:
            capacity - 容量
        
        Returns:
            None
        
        """
        self.buffer=collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加item

        Returns:
            None
        
        """
        self.buffer.append(
            (state, action, reward, next_state, done)
        )
    
    def sample(self, batch_size):
        """
        随机采样

        Parameters:
            batch_size - 采样数目
        
        Returns:
            states - numpy.array

            action - tuple
            
            reward - tuple
            
            next_state - numpy.array
            
            done - tuple
        
        """

        transitions=random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done=\
        zip(*transitions)
        return np.array(state), action, reward,\
    np.array(next_state), done
    
    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        """Q网络"""
        super(Qnet, self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self, x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim,
                lr, gamma, epsilon, target_update, device):
        """

        DQN算法
        
        Parameters:
            state_dim - 状态维度

            hidden_dim - 隐藏层维度
            
            action_dim - 动作种类
            
            lr - 学习率
            
            gamma - 衰减率
            
            epsilon - 贪婪算法的参数
            
            target_update - 目标网络更新频率
            
            device - 设备
        
        Returns:
            None
        
        """

        self.action_dim=action_dim
        # Q网络
        self.q_net=Qnet(state_dim,hidden_dim,
                        self.action_dim).to(device)
        # 目标网络
        self.target_q_net=Qnet(state_dim,hidden_dim,
                        self.action_dim).to(device)
        
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),
                                       lr=lr)
        
        self.gamma=gamma #衰减率
        self.epsilon=epsilon #贪婪策略
        self.target_update=target_update #目标网络更新频率
        self.count=0
        self.device=device
    
    def take_action(self, state):

        """

        根据贪婪算法,基于state采取action

        Parameters:
            state

        Returns:
            action
        
        """

        if random.random() < self.epsilon:
            # 随机
            action=random.randint(0,self.action_dim-1)
        else:
            state = torch.tensor(np.array([state]), 
                                 dtype=torch.float).to(self.device)
            action=self.q_net(state).argmax().item()
        return action
    
    def update(self, transition_dict):

        """

        更新Q网络

        Parameters:
            transition_dict - 采样states, actions, rewards, next_states字典

        Returns:
            None
        
        """


        states=torch.tensor(transition_dict['states'],
                           dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions'],
                        dtype=torch.long).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],
                            dtype=torch.float).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],
                                dtype=torch.float).to(self.device)
        dones=torch.tensor(transition_dict['dones'],
                          dtype=torch.float).view(-1,1).to(self.device)
        
        q_values=self.q_net(states).gather(1,actions)
        
        # 下个状态的最大Q值
        max_next_q_values=self.target_q_net(next_states).max(1).values.view(-1,1)
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)
        
        dqn_loss=torch.mean(F.mse_loss(q_values,q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        
        if self.count%self.target_update==0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count+=1