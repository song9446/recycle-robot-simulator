"""
  TensorFlow translation of the torch example found here (written by SeanNaren).
  https://github.com/SeanNaren/TorchQLearningExample

  Original keras example found here (written by Eder Santana).
  https://gist.github.com/EderSantana/c7222daa328f0e885093#file-qlearn-py-L164

  The agent plays a game of catch. Fruits drop from the sky and the agent can choose the actions
  left/stay/right to catch the fruit before it reaches the ground.
"""

import tensorflow as tf
import tflearn
import numpy as np
import random
import math
import os

import Communicator

# Parameters
epsilon = 0.3  # The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
epsilonMinimumValue = 0.001 # The minimum value we want epsilon to reach in training. (0 to 1)
epoch = 10001 # The number of games we want the system to run for.
maxMemory = 500 # How large should the memory be (where it stores its past experiences).
batchSize = 50 # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
dAction = (3,) # The number of actions. Since we only have left/stay/right that means 3 actions.
dState = (64, 64, 1)
discount = 0.0 # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)  
learningRate = 0.0002 # Learning Rate for Stochastic Gradient Descent (our optimizer).
RANDOM_SEED = 100242

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Create the base model.
X = tf.placeholder(tf.float32, (None,) + dState)
net = tf.contrib.layers.conv2d(X, 64, 3, 2, 
                             activation_fn=tf.nn.relu,
                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                             biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
net = tf.nn.dropout(net, 0.5)
net = tf.contrib.layers.conv2d(net, 64, 3, 1, 
                             activation_fn=tf.nn.relu,
                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                             biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
net = tf.nn.dropout(net, 0.5)
net = tf.contrib.layers.conv2d(net, 64, 3, 1, 
                             activation_fn=tf.nn.relu,
                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                             biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
net = tf.nn.dropout(net, 0.5)
#net = tf.contrib.layers.flatten(net)
                             #kernel_regularizer=tf.nn.l2_loss,
                             #bias_regularizer=tf.nn.l2_loss,)
net = tflearn.fully_connected(net, 1024, activation='relu')
#net = tflearn.fully_connected(net, 300, activation='relu')
# Final layer weights are init to Uniform[-3e-3, 3e-3]
output_layer = tf.contrib.layers.fully_connected(
    net, sum(dAction), activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

# True labels
Y = tf.placeholder(tf.float32, (None,) + dAction)

# Mean squared error cost function
#cost = tf.reduce_sum(tf.square(Y-output_layer)) / (2*batchSize)
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(Y), logits=output_layer))

# Stochastic Gradient Decent Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)



# Helper function: Chooses a random value between the two boundaries.
def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;


# The environment: Handles interactions and contains the state of the environment
class CatchEnvironment():
  def __init__(self, gridSize):
    self.gridSize = gridSize
    self.nbStates = self.gridSize * self.gridSize
    self.state = np.empty(3, dtype = np.uint8) 

  # Returns the state of the environment.
  def observe(self):
    canvas = self.drawState()
    canvas = np.reshape(canvas, (-1,self.nbStates))
    return canvas

  def drawState(self):
    canvas = np.zeros((self.gridSize, self.gridSize))
    canvas[self.state[0]-1, self.state[1]-1] = 1  # Draw the fruit.
    # Draw the basket. The basket takes the adjacent two places to the position of basket.
    canvas[self.gridSize-1, self.state[2] -1 - 1] = 1
    canvas[self.gridSize-1, self.state[2] -1] = 1
    canvas[self.gridSize-1, self.state[2] -1 + 1] = 1    
    return canvas        

  # Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.
  def reset(self): 
    initialFruitColumn = random.randrange(1, self.gridSize + 1)
    initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
    self.state = np.array([1, initialFruitColumn, initialBucketPosition]) 
    return self.getState()

  def getState(self):
    stateInfo = self.state
    fruit_row = stateInfo[0]
    fruit_col = stateInfo[1]
    basket = stateInfo[2]
    return fruit_row, fruit_col, basket

  # Returns the award that the agent has gained for being in the current environment state.
  def getReward(self):
    fruitRow, fruitColumn, basket = self.getState()
    if (fruitRow == self.gridSize - 1):  # If the fruit has reached the bottom.
      if (abs(fruitColumn - basket) <= 1): # Check if the basket caught the fruit.
        return 1
      else:
        return -1
    else:
      return 0

  def isGameOver(self):
    if (self.state[0] == self.gridSize - 1): 
      return True 
    else: 
      return False 

  def updateState(self, action):
    if (action == 1):
      action = -1
    elif (action == 2):
      action = 0
    else:
      action = 1
    fruitRow, fruitColumn, basket = self.getState()
    newBasket = min(max(2, basket + action), self.gridSize - 1) # The min/max prevents the basket from moving out of the grid.
    fruitRow = fruitRow + 1  # The fruit is falling by 1 every action.
    self.state = np.array([fruitRow, fruitColumn, newBasket])

  #Action can be 1 (move left) or 2 (move right)
  def act(self, action):
    self.updateState(action)
    reward = self.getReward()
    gameOver = self.isGameOver()
    return self.observe(), reward, gameOver, self.getState()   # For purpose of the visual, I also return the state.


# The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
# and creates batches of experiences based on the mini-batch size for training.
class ReplayMemory:
  def __init__(self, dState, dAction, maxMemory, discount):
    self.maxMemory = maxMemory
    self.dState = dState 
    self.dAction = dAction 
    self.discount = discount
    canvas = np.zeros(dState)
    self.inputState = np.empty((self.maxMemory,) +  dState, dtype = np.float32)
    self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
    self.nextState = np.empty((self.maxMemory,) + dState, dtype = np.float32)
    self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
    self.rewards = np.empty(self.maxMemory, dtype = np.int8) 
    self.count = 0
    self.current = 0

  # Appends the experience to the memory.
  def remember(self, currentState, action, reward, nextState, gameOver):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.inputState[self.current, ...] = currentState
    self.nextState[self.current, ...] = nextState
    self.gameOver[self.current] = gameOver
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.maxMemory

  def getBatch(self, model, batchSize, sess, X):
    
    # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
    # batch we can (at the beginning of training we will not have enough experience to fill a batch).
    memoryLength = self.count
    chosenBatchSize = min(batchSize, memoryLength)

    inputs = np.zeros((chosenBatchSize,) + self.dState)
    targets = np.zeros((chosenBatchSize,) + self.dAction)

    # Fill the inputs and targets up.
    for i in range(chosenBatchSize):
      if memoryLength == 1:
        memoryLength = 2
      # Choose a random memory experience to add to the batch.
      randomIndex = random.randrange(1, memoryLength)
      current_inputState = np.reshape(self.inputState[randomIndex], (1,) + self.dState)

      target = sess.run(model, feed_dict={X: current_inputState})
      
      current_nextState =  np.reshape(self.nextState[randomIndex], (1,) + self.dState)
      current_outputs = sess.run(model, feed_dict={X: current_nextState}) 
      
      # Gives us Q_sa, the max q for the next state.
      nextStateMaxQ = np.amax(current_outputs)
      if (self.gameOver[randomIndex] == True):
        target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex]
      else:
        # reward + discount(gamma) * max_a' Q(s',a')
        # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
        # to give an error of 0 for those outputs.
        target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

      # Update the inputs and targets.
      inputs[i] = current_inputState
      targets[i] = target

    return inputs, targets


ACTION_MAP = [
    "", "turn_right", "turn_left", "turn_left", "grab", "put"
]
    
def main(_):
  print("Training new model")

  # Define Environment
  world = Communicator.gen_world(1500, 1500)
  #env = CatchEnvironment(gridSize)

  # Define Replay Memory
  memory = ReplayMemory(dState, dAction, maxMemory, discount)

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  
  with tf.Session() as sess:   
    tf.initialize_all_variables().run() 

    for i in range(epoch):
      # Initialize the environment.
      err = 0
      Communicator.init_world(world)
      #env.reset()
     
      isGameOver = False

      # The initial state of the environment.
      currentState = np.dot(Communicator.get_screen_pixels(world, dState[0], dState[1]), [0.299, 0.587, 0.114])[np.newaxis,...,np.newaxis].astype(np.float32)/256.
      lastState = currentState
      #currentState = env.observe()
            
      total_reward = 0
      j = 0
      while (isGameOver != True):
        q=0
        action = -9999  # action initilization
        # Decides if we should choose a random action, or an action from the policy network.
        global epsilon
        if (randf(0, 1) <= epsilon):
          action = random.randrange(1, sum(dAction)+1)
        else:          
          # Forward the current state through the network.
          q = sess.run(output_layer, feed_dict={X: currentState})#np.concatenate((lastState, currentState), axis=3)})          
          # Find the max index (the chosen action).
          index = q.argmax()
          action = index + 1     

        # Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
        if (epsilon > epsilonMinimumValue):
          epsilon = epsilon * 0.999
        
        #nextState, reward, gameOver, stateInfo = env.act(action)
        Communicator.send_action(world, ACTION_MAP[action], 1)
        Communicator.send_action(world, "move_forward", 1)
        nextState = np.dot(Communicator.get_screen_pixels(world, dState[0], dState[1]), [0.299, 0.587, 0.114])[np.newaxis, ...,np.newaxis].astype(np.float32)/256.
        reward = Communicator.get_reward(world)
        Communicator.show(world, 300, 300)
        gameOver = False
        if world.robot.x > world.w or world.robot.x < 0 or world.robot.y > world.h or world.robot.y < 0:
            gameOver = True
        if j>100: gameOver = True
        j+=1
            
        total_reward += reward

        memory.remember(currentState, action, reward, nextState, gameOver)#np.concatenate((currentState,nextState),axis=3), gameOver)
        print(q)
        
        # Update the current state and if the game is over.
        lastState = currentState
        currentState = nextState
        isGameOver = gameOver
                
        # We get a batch of training data to train the model.
        inputs, targets = memory.getBatch(output_layer, batchSize, sess, X)
        
        # Train the network which returns the error.
        _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})  
        err = err + loss

      print("Epoch " + str(i) + ": err = " + str(err) + ": reward = " + str(total_reward))
    # Save the variables to disk.
    save_path = saver.save(sess, os.getcwd()+"/model.ckpt")
    print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  tf.app.run()

