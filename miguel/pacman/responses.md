# Experiment Report: Reinforcement Learning in Pacman-v5

## 1.Describe the selected environment used in the experiment, outlining its main characteristics and objectives.

Initially, the experiments were conducted using **MsPacman-v5** from the Gymnasium Atari suite. After a couple of weeks of trials, it was observed that the MsPacman environment presented significant challenges due to its ghost behavior and maze complexity, which made learning slower and more unstable. As a result, we switched to **Pacman-v5** for the remainder of the experiments.  

The selected environment for this experiment is **Pacman-v5** from the Gymnasium Atari suite.  
- **Objective:** Navigate Pacman through the maze to collect all pellets while avoiding ghosts.  
- **Challenges:**  
  - Multiple ghosts with different movement patterns.  
  - Timeout rules in the original game.  
  - Continuous decision-making required for safe navigation and pellet collection.  
- **Key Features:**  
  - High-dimensional RGB observations of the maze.  
  - Discrete action space representing movement directions (up, down, left, right, no-op, etc.).  
  - Episodic gameplay with multiple lives, normally terminating when Pacman is killed by a ghost or times out.
---

## 2. Describe the preprocessing steps and wrappers applied in your experiment, specifying the resulting observation space.

To facilitate learning with deep reinforcement learning algorithms, the following preprocessing and wrappers were applied:

1. **NoopResetEnv:**  
   Performs a random number of no-op actions at the start of each episode to introduce stochasticity.

2. **GhostLifeEnv:**  
   Ensures Pacman only dies from ghost collisions, ignoring pellet timeout rules.  
   - Allows longer episodes for more stable learning.

3. **MaxAndSkipEnv:**  
   Skips frames to reduce computational load and returns the pixel-wise maximum of the last two frames to handle flickering.

4. **ResizeAndGrayScale:**  
   Converts RGB frames to grayscale and resizes them to 84x84 pixels.  
   - **Resulting observation space:** `(1, 84, 84)` uint8.

5. **ClipRewardEnv:**  
   Clips rewards to the range `{-1, 0, 1}` to stabilize training.

6. **TimeAliveRewardEnv:**  
   Adds a small positive reward for every step Pacman survives (`+0.01` per step) to incentivize exploration and survival.

**Resulting Observation Space: Box(shape=(1, 84, 84), dtype=uint8, low=0, high=255)** 

## 3. Provide a description of the chosen agents or models, including their archi tecture, learning algorithms, and relevant design decisions.

Two reinforcement learning agents were considered:

1. **PPO (Proximal Policy Optimization)**
   - **Architecture:** Convolutional neural network (CNN) with policy and value heads (CnnPolicy).  
   - **Learning Algorithm:** On-policy, gradient-based policy optimization using clipped surrogate objective.  
   - **Design Decisions:**  
     - Frame stacking (4 frames) to capture motion.  
     - Reward normalization for stable advantage estimation.  
     - Learning rate and other hyperparameters optimized via Optuna.

2. **A2C (Advantage Actor-Critic)**
   - **Architecture:** Similar CNN backbone with actor-critic heads.  
   - **Learning Algorithm:** Synchronous advantage actor-critic, using n-step returns for bootstrapping.  
   - **Design Decisions:**  
     - Normalized advantages.  
     - Entropy coefficient tuned for exploration.

---

## 4. Train the models, applying fine-tuning or hyperparameter optimization when necessary to improve performance.

- **Training Procedure:**  
  - Vectorized environments with 4 parallel instances.  
  - Total timesteps: configurable, e.g., 10 million steps for PPO.  
  - Callback logging with WandB for episodic rewards and lengths.  
- **Hyperparameter Optimization:**  
  - Optuna used to tune learning rate, n_steps, gamma, entropy coefficient, and other algorithm-specific parameters.  
  - Pruning enabled to stop unpromising trials early.  

- **Fine-Tuning:**  
  - Batch size adjusted to ensure compatibility with n_steps (for PPO).  
  - Reward clipping and time-alive reward used to stabilize learning.

---

## 5. Save the best-performing models and produce visualizations that illustrate their learning curves and in-environment behavior.
## 6. Evaluate the trained models within the environment, reporting their average success rate and any relevant performance metrics.
## 7. Present the results, analyze the findings, and justify the selection of the model that achieves the best performance for this environment.



