import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os


class Qnetwork():
    def __init__(self):
        #These lines establish the feed-forward part of the network used to choose actions
	#The game state is a 19-element vector, fed into two fully connected layers
	#Output is Q-values for each possible action that can be taken
        self.sz1 = 30 #Input into this size of layer
        self.sz2 = 19 #Into this size layer
        self.act_space = 6 #Output is the size of the action space
        self.inputs1 = tf.placeholder(shape=[None,19],dtype=tf.float32)
        self.l1 = tf.Variable(tf.random_uniform([19,self.sz1],-.1,.1))
        self.b1 = tf.Variable(tf.random_uniform([1,self.sz1],-.1,.1))
        self.l2 = tf.Variable(tf.random_uniform([self.sz1,self.sz2],-.1,.1))
        self.b2 = tf.Variable(tf.random_uniform([1,self.sz2],-.1,.1))
        self.l3 = tf.Variable(tf.random_uniform([self.sz2,self.act_space],-.1,.1))
        self.b3 = tf.Variable(tf.random_uniform([1,self.act_space],-.1,.1))
        self.int1 = tf.tanh(tf.matmul(self.inputs1,self.l1) + self.b1 )
        self.int2 = tf.tanh(tf.matmul(self.int1,self.l2) + self.b2 )
        self.Qout = tf.matmul(self.int2,self.l3) + self.b3
        self.predict = tf.argmax(self.Qout,1) #Maximum of the Q-values decides what action is best
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.act_space,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

#Without a buffer, the network only learns the current state and can "chase its own tail" in a way
#Storing a buffer of past actions and sampling many points randomly from it for training allows more stable convergence
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

#Even then, convergence can be poor. A two network system works better, with a fast network making decisions that slowly update
#a slower network that keeps it in check, giving much more stable learning. This function updates the slow network from the fast network.
#the update structure lets the fast network make immediate reward decisions, but the slow network keeps track of the future reward.
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

#The game has 4 locations on a 5x5 grid a passenger can be or need to go. The taxi is controlled and moves around, though there are walls
#Every move in a cardinal direction loses a point, and 10 points are lost for trying to pick up a passenger where they are not
# or trying to drop off a passenger without a passenger in the car or trying to drop a passenger outside of a possible goal location
# no points are lost when a passenger is validly picked up or placed in a possible goal, and 20 points are won when the passenger is 
# correctly brought to the goal.

def interpret(u):
    destidx = u % 4
    u = u // 4
    passloc = u % 5
    u = u // 5
    taxicol = u % 5
    u = u // 5
    taxirow = u

    s = [0] * 19
    s[0:5] = id5[taxirow] * 2 - 1
    s[5:10] = id5[taxicol] * 2 - 1
    s[10:15] = id5[passloc] * 2 - 1
    s[15:19] = id4[destidx] * 2 - 1
    return s

env = gym.make('Taxi-v2') #Set up the game to learn

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .98 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 120000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 200 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = os.path.dirname(os.path.abspath(__file__)) + '/dqn_30_19_layer_short_run' #The path to save our model to.
tau = 0.001 #Rate to update target network toward primary network
id4 = np.identity(4)
id5 = np.identity(5)

tf.reset_default_graph()
mainQN = Qnetwork()
targetQN = Qnetwork()

init = tf.initialize_all_variables()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0
succ_count = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps,sess) #Update the slower target network with fast primary network
    for i in range(num_episodes):

        if i > 0 and i % 1000 == 0: #Play 100 games and report how well it did
            win_count = 0
            score_list = []
            for h in range(100):
                u = env.reset()
                s = interpret(u)
                d = False
                score = 0
                
                j = 0
                while j < max_epLength:
                    j += 1
                    a = sess.run(mainQN.predict,feed_dict={mainQN.inputs1:[s]})[0]
                    u,r,d,_ = env.step(a)
                    score += r
                    s = interpret(u)
                    if d == True:
                        win_count += 1
                        break
                score_list.append(score)
            with open(path + '/test_results'+str(i)+'.txt', 'w') as f:
                f.write('The scores for each game at training iteration '+str(i)+' are: ')
                f.write(', '.join(str(x) for x in score_list)+'\n')
                f.write('The average score for the 100 games is: '+str(np.mean(score_list)))

        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        u = env.reset()
        s = interpret(u)
        d = False
        rAll = 0
        j = 0
        #The Q-Network is trained here to evaluate the discounted future rewards of making particular moves in the game
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = env.action_space.sample()
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.inputs1:[s]})[0]
            u1,r,d,_ = env.step(a)
            s1 = interpret(u1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r/50.0,s1,d]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    #The networks train on a random batch pulled from the buffer
                    #the buffer covers the reward for performing the action in the buffer.
                    #both networks look forward, and the fast network chooses what move it would make next.
                    #the slow network's evaluation of the future reward is used to update the fast network.
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.inputs1:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.inputs1:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.inputs1:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
            rAll += r
            s = s1
            
            if d == True:
                succ_count += 1
                break
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print "Saved Model"
        if len(rList) % 10 == 0:
            print total_steps,np.mean(rList[-10:]), e,i,succ_count
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
print "Percent of succesful episodes: " + str(succ_count // num_episodes) + "%"

rMat = np.resize(np.array(rList),[len(rList)/100,100])
rMean = np.average(rMat,1)
