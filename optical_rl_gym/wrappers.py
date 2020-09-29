import gym


class UseInfoReward(gym.RewardWrapper):

    def __init__(self, env, info_key):
        self.env = env
        self.info_key = info_key
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, info), done, info

    def reward(self, reward, info):
        return info[self.info_key]
