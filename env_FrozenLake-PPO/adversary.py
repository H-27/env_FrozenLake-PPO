import numpy as np
import Actor
import Critic
from PPO_Agent import PPO_Agent
from Buffer import Buffer
import FrozenLake
import tensorflow as tf
class Adversary(object):

    def __init__(self, env, training_epochs, epoch_length):
        self.env = env
        self.n_training_epochs = training_epochs
        self.protagonist = PPO_Agent(4, epoch_length)
        self.antagonist = PPO_Agent(4, epoch_length)
        self.n_steps = epoch_length
        self.memory = Buffer()
        self.gamma = 0.95
        self.action_space = 5 # 0: place start, 1: place goal, 2: place hole, 3: do nothing, 4: place obstacle
        self.adversary = PPO_Agent(5, epoch_length)
        

    def adversarial_epoch(self, env):
        # we initialize the map with a start and goal in case the adversary 
        #misses these actions in the beginning of the training
        env.reset_map()
        # loop through all possible coordinates and let network decide on the action
        print('Creating Map.')
        for i in range(1, env.x-1):
            for j in range(1, env.y-1):
                one_hot_map = env.draw_for_state()
                # set coordinate temporary to 'P' so adversary knows his placement
                one_hot_map[i,j] = b'P'
                one_hot_map = env.one_hot(one_hot_map)
                action, probs = self.adversary.get_action(one_hot_map)
                value = self.adversary.critic(np.array([one_hot_map]), training = False)
                # change coordinate according to chosen action
                env.map[i,j] = self.choose_action(action)
                state = env.one_hot(env.map.copy())

                if i == env.x-1 and j == env.y-1:
                    done = True
                else:
                    done = False
                self.memory.storeTransition(state, action, 0, value[0][0], probs[0], done)
        print('Map created.')
        print('Training Protagonist.')
        # run protagonist through env and train, the get reward
        for _ in range(self.n_training_epochs):
            self.train_agent(env, self.protagonist, self.n_steps)
        protagonist_reward = [self.get_performance(env, self.protagonist) for _ in range(5)]
        protagonist_reward = np.mean(protagonist_reward, dtype= np.float32)
        print('Training Antagonist.')
        # run antagonist through env and train
        for n in range(self.n_training_epochs):
            self.train_agent(env, self.antagonist, self.n_steps)
        antagonist_reward = [self.get_performance(env, self.antagonist) for _ in range(5)]
        antagonist_reward = np.array(antagonist_reward, dtype = np.float32)
        antagonist_reward = np.max(antagonist_reward)

        # calculate regret
        regret = tf.subtract(antagonist_reward, protagonist_reward)
        print(regret)
        print('Training Adversary')

        self.adversary.learn_with_regret(self.memory, regret)
        return regret.numpy()#, total_loss


    def choose_action(self, action):
        # 0: place start, 1: place goal, 2: place hole, 3: do nothing, 4: place obstacle
        if action == 0:
            return b'S'
        if action == 1:
            return b'G'
        if action == 2:
            return b'H'
        if action == 3:
            return b'-'
        if action == 4:
            return b'x'
    
    def train_agent(self, env, agent, episode_length: int):
        done = False
        env.reset()
        position = env.reset
        episode_values = []
        episode_dones = []
        episode_rewards = []
        agent_memory = Buffer()
        
        c = 0
        max_steps_episode = (env.x-2)*(env.y-2) * 3
        while c <= episode_length:
            if max_steps_episode == 0:
                print('Agent stuck')
                break
            
            observation = env.draw_for_state()
            observation = env.one_hot(observation)
            action, probs = agent.get_action(observation)
            value = agent.critic(np.array([observation]), training = False).numpy()

            episode_values.append(value[0][0])
            next_position, reward, done = env.step(action)

            episode_dones.append(done)
            episode_rewards.append(reward)

            agent_memory.storeTransition(observation, action, reward, value[0][0], probs[0], done)

            if done:
                env.reset()
                value = agent.critic(np.array([observation]), training = False).numpy()
                episode_values.append(value)
                c +=1 
                d_returns = agent_memory.calculate_disc_returns(episode_rewards, self.gamma)
                adv = agent_memory.calculate_advantage(episode_rewards, episode_values, episode_dones, self.gamma)

                episode_values = []
                episode_dones = []
                episode_rewards = []
            max_steps_episode -= 1

        for epochs in range(10):
            actor_loss, critic_loss = agent.learn(agent_memory)
        agent_memory.clear()


    def get_performance(self, env, agent):

        env.reset()
        done = False
        n_steps = 0
        t = (env.x-2) * (env.y - 2) # max episode length
        while not done:
            observation = env.draw_for_state()
            observation = env.one_hot(env.map)
            action, _ = agent.get_action(observation)
            n_steps += 1
            _, reward, done = env.step(action)
        #return np.power(self.gamma, n_steps) * reward
        return 1 - 0.9 * (n_steps/t)

