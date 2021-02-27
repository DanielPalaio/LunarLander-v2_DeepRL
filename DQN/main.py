

from agent import Agent
import gym


env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
train = 0
test = 1
num_episodes = 100
graph = True

file_type = 'tf'
file = 'saved_networks/dqn_model104'

dqn_agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=4, epsilon=1.0, batch_size=64, input_dims=8)

if train and not test:
    dqn_agent.train_model(env, num_episodes, graph)
else:
    dqn_agent.test(env, num_episodes, file_type, file, graph)
