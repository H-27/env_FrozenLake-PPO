from PPO_Agent import PPO_Agent
from Buffer import Buffer
from FrozenLake import FrozenLake
import numpy as np
import gym
import sys
import tensorflow as tf

class PPO(object):
    def __init__(self, action_space, episode_length, gamma: float = 0.95):

        self.episode_length = episode_length
        self.agent = PPO_Agent(action_space, episode_length)
        self.buffer = Buffer()
        self.episode_reward = []
        self.total_average = []
        self.target = False 
        self.best_reward = 0
        self.avg_rewards_list = []
        self.gamma = 0.95

    def test_reward(self, env):
        total_reward = []
        env.reset()
        done = False
        observation = env.one_hot(env.map)


        while not done:
            
            position = np.array([env.position])
            action, _ = self.agent.get_action(observation)
            next_position, reward, done = env.step(action)
            total_reward.append(reward)

        return np.mean(total_reward)

    def run(self, env, episode_number: int):
        counter = 0
        env.reset
        observation = np.array([env.one_hot(env.map)])
        
        while counter <= episode_number:

            print('Starting Episode:' + str(counter+1))         
            done = False
            self.buffer.clear()
            env.reset()
            position = env.reset()
            episode_values = []
            episode_dones = []
            episode_rewards = []
            
            c = 0
            while c <= self.episode_length:
                
                observation = env.draw_for_state()
                observation = env.one_hot(observation)
                
                #print(position)
                action, probs = self.agent.get_action(observation)
                value = self.agent.critic(np.array([observation])).numpy()

                episode_values.append(value[0][0])
                next_position, reward, done = env.step(action)

                episode_dones.append(done)
                episode_rewards.append(reward)

                self.buffer.storeTransition(observation, action, reward, value[0][0], probs[0], done)

                if done:
                    env.reset()
                    value = self.agent.critic(np.array([observation])).numpy()
                    episode_values.append(value)
                    c +=1 
                    d_returns = self.buffer.calculate_disc_returns(episode_rewards, self.gamma)
                    adv = self.buffer.calculate_advantage(episode_rewards, episode_values, episode_dones, self.gamma)

                    episode_values = []
                    episode_dones = []
                    episode_rewards = []

                    
            print('Training')
            for epochs in range(10):
                actor_loss, critic_loss = self.agent.learn(self.buffer)

            counter+=1
