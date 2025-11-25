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
The best performing models are saved on the folder ```
checkpoints/```, you can find some visualizations of the training logs and agents playing in ```
docs/plots/rewards_vs_frames/``` and ```docs/plots/matches/``` respectively. 

**5. Evaluate the trained models within the environment, reporting the average success rate.** 
To evaluate the generalization and stability of our agents, we ran a dedicated test phase for 50 episodes using a purely greedy policy (epsilon=0). Success is defined as the agent winning the match, achieving a positive total reward.
Evaluation Results Summary:
N-step DQN: 
Average reward over 50 episodes: 11.00
Success rate: 100.00%

Base DQN: 
Average reward over 50 episodes: 21.00
Success rate: 100.00%

Double DQN:
Average reward over 50 episodes: 21.00
Success rate: 100.00%

Dueling DQN:
Average reward over 50 episodes: 21.00
Success rate: 100.00%

The evaluation confirms that all implemented models converged to a winning policy, achieving a 100% success rate over the 50 test episodes. This means that every agent successfully won the game against the fixed opponent policy.
However, a significant difference exists in the Average Reward:
The Base DQN, Double DQN, and Dueling DQN models consistently achieved the maximum possible reward (Avg Reward +21.00), implying they won nearly every match with a perfect score (e.g., 21-0). This demonstrates that these policies are highly robust and have identified a deterministic optimal strategy that the opponent cannot counter.
The N-Step DQN agent, while still winning every match, scored an average reward of +11.00 (implying a winning score closer to 21-10). This indicates that the N-Step agent converged to a policy that is sub-optimal in terms of points conceded. This might be due to the higher variance introduced by the multi-step updates during training, preventing it from perfecting the final stages of the opponent's exploitation, despite the faster convergence speed observed during the training logs.
The results confirm that the Double DQN and Dueling DQN architectures not only achieved high convergence speed (as seen in the training logs) but also obtained the perfect score, so the highest result.

**6. Discuss the obtained results, analyze the findings, and justify the selection of the best model for this environment.**
The analysis of the training logs reveals a clear distinction in sample efficiency between the standard Deep Q-Network and its extensions. Specifically, the Double DQN, N-Step DQN, and Dueling DQN models demonstrated significantly faster convergence, requiring approximately 268,000 to 275,000 frames to reach the target reward threshold of +19. In contrast, the Basic DQN was considerably slower, needing around 392,000 frames to achieve the same level of performance. Among the variants, the Double DQN and N-Step DQN proved to be the fastest to converge during the initial training phase.
The evaluation phase confirmed that while all agents achieved a 100% success rate, only the Base, Double, and Dueling models attained the maximum possible reward of +21.00. The N-Step DQN, conversely, yielded a lower average of +11.00, indicating a winning but sub-optimal policy compared to the deterministic perfection of the other architectures.
Based on the combined evidence of convergence speed and final policy quality, we conclude that the Double DQN is the best model for solving this environment. While the Dueling DQN also achieved a perfect score and rapid training, the Double DQN matched this high-quality policy while offering the most efficient stability against Q-value overestimation. The decoupling of action selection and evaluation proved to be the decisive mechanism, allowing the agent to achieve optimal performance with high sample efficiency, making it the most reliable choice for the Pong environment.






