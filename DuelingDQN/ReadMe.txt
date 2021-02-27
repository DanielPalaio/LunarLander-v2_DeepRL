Trained on a NVIDIA GeForce MX 250

env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")

num_episodes=500
lr=0.00075
discount_factor=0.99
num_actions=4
epsilon=1.0
batch_size=64
input_dim=[8]
update_rate=120

Test - 'saved_networks/duelingdqn_model123'