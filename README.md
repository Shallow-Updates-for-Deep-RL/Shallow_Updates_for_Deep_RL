# Shallow_Updates_for_Deep_RL

Official implementation for the paper: "Shallow Updates for Deep Reinforcement Learning"

![ScreenShot](alg_diag.png =800x800)

This code extends the official DQN implementation [[link](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)].

To run:
1. Follow the official isntallation isntructions.
2. Replace NeuralQLearner.lua with our code (NeuralQLearnerShallow.lua)
3. Change initenv.lua, line 14: from "require 'NeuralQLearner'" to "require 'NeuralQLearnerShallow'".

Features supported:
1. Netowrk type: DQN, and DDQN (line 155).
2. SRL method: LSTD-Q, FQI, or none (line 156).
3. Regularizer coefficient for the SRL methods (line 157).
4. SRL updates frequency (line 158).
