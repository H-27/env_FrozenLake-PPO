class Main(object):
    if __name__=="__main__":
        #tf.compat.v1.disable_eager_execution()
        env = FrozenLake(5,5)

        env.map[3,3] = b'H'
        env.map[3,4] = b'H'
        env.map[3,5] = b'H'
        # p = PPO(4, 3)
        # p.run(env, 3, 3)
        env.reset()
        algo = Adversary(env, 2, 3)
        print(algo.adversarial_epoch(env))