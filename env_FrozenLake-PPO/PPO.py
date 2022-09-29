from PPO_Agent import PPO_Agent
from Buffer import Buffer
from FrozenLake import FrozenLake
import numpy as np
import gym
import sys
import tensorflow as tf

class PPO(object):
    def __init__(self, action_space, gamma: float = 0.95):

        self.agent = PPO_Agent(action_space)
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
        observation = np.array([env.one_hot(env.map)])


        while not done:
            
            position = np.array([env.position])
            action, _ = self.agent.get_action(observation)
            next_position, reward, done = env.step(action)
            total_reward.append(reward)

        return np.mean(total_reward)

    def run(self, env, episode_number: int, episode_length: int):
        
        target = False
        best_reward = 0
        self.agent.old_probs = [[0.25, 0.25, 0.25, 0.25, 0.25] for i in range(episode_length)]
        counter = 0
        env.reset
        observation = np.array([env.one_hot(env.map)])
        
        while counter <= episode_number:

            print('Starting Episode:' + str(counter+1))         
            if target == True:
                break
            done = False
            self.buffer.clear()
            env.reset()
            position = env.reset
            episode_values = []
            episode_dones = []
            episode_rewards = []
            
            c = 0
            while c <= episode_length:
                
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
                    print(c)
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
        
            ##rewards = [self.test_reward(env) for _ in range(10)]
            #avg_reward = self.test_reward(env)#np.mean(rewards)
            #best_reward = 0
            #print('Episode: ' + str(counter))
            #print(f"total test reward is {avg_reward}")
            #if avg_reward > best_reward:
            #    print('best reward=' + str(avg_reward))
            #    algo.agent.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            #    algo.agent.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            #    best_reward = avg_reward
            #if best_reward*100 == 75:
            #    target = True
            #env.reset_agent(0)
            #state = env.agent_pos[0] #env.reset()
            #observation = state[0]* env.nrow+ state[1]
            #for i in range(4):#env.observation_space.n):
            #    print('probs for state:' + str(i))
            #    print(self.agent.actor.predict(np.array([one_hot_states[observation]])))
            counter+=1
