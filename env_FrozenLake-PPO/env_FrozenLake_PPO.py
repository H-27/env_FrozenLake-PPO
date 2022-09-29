from FrozenLake import FrozenLake
from PPO import PPO
if __name__=="__main__":
    env = FrozenLake(3,3)
    env.reset()
    algo = PPO(4)
    
    algo.run(env, 500, 1)
    algo.test_reward(env)