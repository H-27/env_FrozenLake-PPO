import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from Actor import Actor
from Critic import Critic
from Buffer import Buffer
from Actor_LSTM import ActorLSTM
from Critic_LSTM import CriticLSTM

class PPO_Agent(object):

    def __init__(self, number_actions: int, episode_length, alpha: float = 0.0001, gamma: float = 0.95, epsilon: float = 0.25):
        self.n_actions = number_actions
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actor = Actor(number_actions, 16, 32, 16)
        self.critic = Critic( 16, 32, 16)
        self.regret_actor = ActorLSTM(4).compile(optimizer="Adam")
        self.regret_critic = CriticLSTM().compile(optimizer="Adam")
        l_inputs = [0.25 for n in range(number_actions)]
        self.old_probs = [l_inputs for i in range(episode_length)]

    def get_action(self, observation):
        
        observation = np.array([observation])
        action_probs = self.actor(observation, training = False)
        action_probs = action_probs.numpy()
        probs = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
        action = int(probs.sample())
        
        return action, action_probs

    def calculate_loss(self, probs, actions, adv, old_probs, critic_loss):
        
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs,tf.math.log(probs))))
        sur1 = []
        sur2 = []
        actor_loss = []
        
        for pb, t, op, a  in zip(probs, adv, np.squeeze(old_probs), actions):
                        t =  tf.constant(t)
                        pb = np.array(pb)
                        tf.stop_gradient(pb)
                        ratio = tf.math.divide(pb[a],op[a])
                        s1 = tf.math.multiply(ratio,t)
                        s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon),t)
                        actor_loss.append(tf.math.minimum(s1,s2))
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) + critic_loss - 0.001 * entropy)
        return actor_loss, loss

    def learn(self, memory):
        
        states = np.array(memory.states, dtype=np.float32)    

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor((states), training=True)
            v = self.critic((states),training=True)
            v = tf.reshape(v, (len(v),))
            c_loss = 0.5 * kls.mean_squared_error(memory.discounted_returns,v)
            a_loss, total_loss = self.calculate_loss(p, memory.actions, memory.advantage, self.old_probs, c_loss)
        
        grads1 = tape1.gradient(total_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)

        self.optimizer_actor.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss#, total_loss

    def learn_with_regret(self, memory, regret):
        regret = tf.convert_to_tensor(regret)
        states = np.array(memory.states)
            self.regret_actor.fit(states, regret)
        self.regret_critic.fit(states, regret)
        
