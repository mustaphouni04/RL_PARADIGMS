### Questions (Step 1)

**1. Describe the preprocessing steps and wrappers applied in your experiment, specifying the resulting observation space.**

We'll apply a chain of wrappers to the environment to allow for compatibility during training:
* **FireResetEnv**: It handles cases in the environment that require the "FIRE" action to start the game. It automatically executes the "FIRE" actions (Actions 1 and 2) to start the game when the environment is reset. If the game ends after executing the action, the environment is reset again ensuring the environment is playable.

* **MaxAndSkipEnv**: It reduces time step size and computational burden by executing the action once every 4 skipped frames (skip=4). It maximizes the observations in skipped frames to reduce flickering and accumulates the total reward during skipped frames. It aids in improving training efficiency and maintaining visual continuity in the game.

* **ProcessFrame84**: Preprocesses image observations by converting the RGB images to grayscale images (using the brightness formula) and resizes the image from 210×160 or 250×160 to 84×84 cropping the image region (from 110×84 to 84×84), removing irrelevant information such as scoreboards. It ends up normalizing the observation space to an 84×84×1 grayscale image.

* **BufferWrapper**: Creates frame stacks to provide temporal information. It stacks multiple consecutive frames together as a single observation. It enables the agent to perceive motion and temporal dynamics and maintain a frame buffer, updating the oldest frame on each observation. It initializes the buffer on reset.

* **ImageToPyTorch**: Adjusts the image dimension format to fit PyTorch. Changes the image format from HWC (height × width × channels) to CHW (channels × height × width) since PyTorch convolutional layers expect channel dimensions to be first. Furthermore, it updates the observation space shape to reflect the new dimension order.

* **ScaledFloatFrame**: Normalize pixel values. Scale pixel values from the integer range [0, 255] to the floating-point range [0.0, 1.0]. Improving numerical stability helps neural network training convergence. Converting data types to float32.

 
All of these wrappers are chained in the following order changing the how the observation is perceived:

Standard Env.        : (210, 160, 3) - the standard environment giving the raw RGB images

MaxAndSkipEnv        : (210, 160, 3) - execute action every n skipped frames, no change to observation

FireResetEnv         : (210, 160, 3) - again, handling actions, so no change to the environment

ProcessFrame84       : (84, 84, 1) - resize the frame and convert it to grayscale, reducing the number of channels

ImageToPyTorch       : (1, 84, 84) - reshape the image and convert it to suitable format for training in PyTorch

BufferWrapper        : (4, 84, 84) - create frame stacks, 4 frames per observation

ScaledFloatFrame     : (4, 84, 84) - scale value ranges for training stability

**2. Provide a detailed description of the selected agents or models.**

For this exercise we selected a standard Deep Q-Network (DQN) as our baseline and compared its performance against three specific extensions: N-Step DQN, Double DQN, and Dueling DQN. 
All models share the same underlying Convolutional Neural Network (CNN) architecture for processing the input frames, known as the "Nature CNN," which consists of three convolutional layers followed by fully connected layers.

#### Baseline: Deep Q-Network (DQN)
Our baseline model is the Deep Q-Network, which combines Q-learning with deep neural networks to approximate the optimal action-value function. This model maps state-action pairs to values using non-linear functions. Our implementation represents a naïve basic DQN that uses the same network for both the current and next states. However, we incorpore the Experience Replay Buffer, so instead of learning from data sequentially, which creates high correlations between consecutive frames, we store transitions (state, action, reward, next state) in a fixed-size buffer. The network trains on a random batch of these stored experiences, which breaks the temporal correlation of the data and stabilizes training.
#### Extension 1: N-Step DQN
The first extension addresses the limitation of the standard DQN, which relies on a single-step update based on the immediate reward and the estimated value of the next state. This can lead to slow learning in environments where rewards are sparse.
In our N-Step implementation, we modify the update rule to look ahead more than one step, particularly 3. We unroll the Bellman equation to accumulate the actual rewards observed over N consecutive steps before using the target network to estimate the value of the final state in the sequence. By using this approach, we can significantly speed up the propagation of rewards and improve convergence speed without introducing too much variance, which could occur with larger N values.
#### Extension 2: Double DQN (DDQN)
We implemented Double DQN to solve the problem of overestimation bias found in the basic architecture. In standard DQN, the same network is used to both select the best action and estimate its value during the target calculation, which often leads to optimistic value estimates.
Double DQN decouples these two processes. In our implementation, we use the Online Network (the one being trained) to determine which action maximizes the value in the next state. However, we use the Target Network to calculate the actual Q-value of that specific chosen action. This simple modification in the loss calculation logic effectively mitigates overestimation and leads to more stable policies.
#### Extension 3: Dueling DQN
Finally, we implemented the Dueling DQN architecture. This extension changes the structure of the neural network itself to better reflect the structure of the problem. The core insight is that for many states, it is unnecessary to estimate the value of each action individually to understand the value of the state itself.
After the convolutional layers extract features, our network splits into two separate streams:
1.	Value Stream: Estimates the scalar value of the current state, representing how good it is to be in that state regardless of the action taken.
2.	Advantage Stream: Estimates the advantage of each specific action compared to the others.
These two streams are then aggregated in the final layer to produce the final Q-values. To ensure the model is identifiable, we subtract the mean of the advantage values during this aggregation process. This architecture improves convergence speed and stability by allowing the agent to learn the state value efficiently.


**3. Train the models, applying fine-tuning or hyperparameter optimization as needed.**
The models are trained and the results are shown in the following exercises.

**4. Save the best-performing models and produce visualizations of their behavior.**
The best performing models are saved on the folder ```shell 
checkpoints/```, you can find some visualizations of the training logs here:




