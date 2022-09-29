from FrozenLake import FrozenLake
from PPO import PPO
from adversary import Adversary
if __name__=="__main__":
    #env = FrozenLake(3,3)
    #env.reset()
    #algo = PPO(4, 2)
    
    #algo.run(env, 4)
    #algo.test_reward(env)
        env = FrozenLake(5,5)

        env.map[3,3] = b'H'
        env.map[3,4] = b'H'
        env.map[3,5] = b'H'
        # p = PPO(4, 3)
        # p.run(env, 3, 3)
        env.reset()
        algo = Adversary(env, 2, 3)
        print(algo.adversarial_epoch(env))