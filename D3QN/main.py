

from agent import Agent
import numpy as np
import gym


class DiscreteActionSpaceWrapper(gym.Wrapper):
    def __init__(self, environment):
        action_space = environment.action_space
        assert isinstance(action_space, gym.spaces.Box)

        environment.action_space = gym.spaces.Discrete(8)
        super(DiscreteActionSpaceWrapper, self).__init__(environment)

    def step(self, discrete_action):
        continuous_action = {
            0: np.array([-1, 0]),
            1: np.array([0.2, 0]),  # main 60%
            2: np.array([0.6, 0]),  # main 80%
            3: np.array([1, 0]),  # main 100%
            4: np.array([-1, -0.75]),  # left 75%
            5: np.array([-1, -1]),  # left 100%
            6: np.array([-1, 0.75]),  # right 75%
            7: np.array([-1, 1])  # right 100%
        }[discrete_action]

        obs, reward, done, info = self.envirnoment.step(continuous_action)
        return obs, reward, done, info


env = DiscreteActionSpaceWrapper(gym.make("LunarLanderContinuous-v2"))
spec = gym.spec("LunarLanderContinuous-v2")
train = 1
test = 0
num_episodes = 350
graph = True

file_type = 'tf'
file = 'saved_networks/d3qn_model5'

d3qn_agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=8, epsilon=1.0, batch_size=64, input_dim=[8])

if train and not test:
    d3qn_agent.train_model(env, num_episodes, graph)
else:
    d3qn_agent.test(env, num_episodes, file_type, file, graph)
