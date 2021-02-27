# OpenAI LunarLander-v2 DeepRL-based solutions
Using Deep Q-Network (DQN), Dueling DQN, and Dueling Double DQN (D3QN)

# Software/Requirements
Module | Software/Hardware
------------- | -------------
Python IDE | Pycharm
Deep Learning library | Tensorflow + Keras
GPU | GeForce MX 250
Interpreter | Python 3.8
Python Environment | Anaconda
Packages | requirements.txt

**To setup Pycharm + Anaconda + GPU, consult the setup file [here](setup.txt)**.  
**To import the required packages, [requirements.txt](DQN/requirements.txt), type the following instruction in the project environment terminal:**  
> pip install -r requirements.txt

# :warning: **WARNING** :warning:  
The training generates a [.txt file](DQN/saved_networks.txt) that tracks the network models (in 'tf' and .h5 formats) that achieved the solved requirement of the environment. Additionally, an overview image (graph) of the training procedure is created.   
Keep in mind that to perform several training processes, the .txt, .png, and directory names must be change. Otherwise, information of previous trainings will get overwritten, and lost.  

Regarding testing, if you choose to load the .h5 model, a 5 episode training is done to initialize/build the keras.model network. Thus, the warnings above mentioned are also appliable to this situation.   
Loading the saved model in 'tf' is the recommended option. After finishing the testing, an overview image (graph) of the training procedure is also generated.

# OpenAI LunarLander-v2
**Actions:**<br />
0 - No action  
1 - Fire left engine  
2 - Fire main engine  
3 - Fire right engine  

**States:**<br />
0 - Lander horizontal coordinate  
1 - Lander vertical coordinate  
2 - Lander horizontal speed  
3 - Lander vertical speed  
4 - Lander angle  
5 - Lander angular speed  
6 - Bool: 1 if first leg has contact, else 0  
7 - Bool: 1 if second leg has contact, else 0  

**Rewards:**<br />
Moving from the top of the screen to the landing pad with (0 velocity) gives a scalar reward between 100 and 140    
Negative reward if the lander moves away from the landing pad   
If the lander crashes, a scalar reward of -100 is given  
If the lander comes to rest, a scalar reward of 100 is given  
Each leg with ground contact corresponds to a scalar reward of 10  
Firing the main engine corresponds to a scalar reward of -0.3 per frame   
Firing the side engines corresponds to a scalar reward of -0.3 per frame   

**Episode termination:**<br />
Lander crashes  
Lander comes to rest  
Episode length is greater than 400  

**Solved Requirement:**<br />
Average reward of 200.0 over 100 consecutive trials

# Deep Q-Network (DQN)
<p align="center">
  <img width="950" height="500" src="https://user-images.githubusercontent.com/79323290/109340337-cb6c5480-7860-11eb-9411-42b8e0d5941d.png">
</p>
       
