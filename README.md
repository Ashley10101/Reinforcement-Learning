# Reinforcement-Learning

This is used to record learning Reinforcement learning, reference code:
1. https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

## Content
### 1. Q-learning  (off-policy)
#### 1.1 Treasure_on_right
`---o---T`  
**state**: -; o (current state)    
**action**: left; right  
**T**: target  
**environment reward**: target=1; others=0  
#### 1.2 Maze
![GitHub Logo](1_Q-learning/Maze/img.png)  
**action**: up; down; right; left   
**Red rectangle**: explorer  
**Black rectangles**: hells&nbsp;&nbsp;[reward = -1].      
**Yellow bin circle**: paradise&nbsp;&nbsp;[reward = +1].      
**All other states**: ground&nbsp;&nbsp;[reward = 0].  

### 2. Sarsa (on-policy)
### 3. Sarsa-lambda (on-policy)
### 4. Deep Q Network (off-policy)
Q-learning + Neural network  
#### advantages:
Q-table usually can only store very limited states, and searching through the form is very time consuming. But using deep Q network, it can be a good way to alleviate these problems.  
  
s -> NN -> actions value  
#### Key points:  
(1) Experience replay  
(2) Fixed Q-targets  
#### Steps:
`env`:   
tensorflow 2.14.0  
python 3.8  
#### 4.1 Maze
![GitHub Logo](1_Q-learning/Maze/img.png) 
#### 4.2 cartpole
![GitHub Logo](3_DQN/CartPole/img.png)   
ref: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
#### 4.3 Mountain Car  
![GitHub Logo](3_DQN/MountainCar/img.png) 
ref: https://www.gymlibrary.dev/environments/classic_control/mountain_car/   
### 5. Policy Gradient (off-policy)  
directly output the action, rather than its value
existing problem:  
(1) probability of unsampled action decrease  
(2) can only update the parameters after the entire trajectory
### 6. Actor Critic
advantages:  
(1) can update the parameters using single step's information  
existing problem:  
(1) it's hard for Critic to converge