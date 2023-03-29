import numpy as np
import pandas as pd
import csv
import random
import torch

'''
THIS FILE IS NULL!!!   
'''

class RL:
    def __init__(self, x):
        self.actionSpace = [0, 1, 2, 3]
        self.stateSpace = x
        self.criterion = torch.nn.CrossEntropyLoss()

    def actionSpaceSample(self):
        return np.random.choice(self.actionSpace)

    def loss(self, y, action):
        Action = [1.0, 0, 0, 0]
        #Action[action] = 1.0
        lossValue = self.criterion(torch.tensor([Action]), y)
        
        return -1 * lossValue

def TrainingDataLoading():
    
    subjects = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8]
    versions = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1]
    xtime_main = []
    x_main = []
    y_main = []
    for i in range(len(subjects)):
        xtime_df = pd.read_csv('TrainingData\subject_00'+str(subjects[i])+'_0'+str(versions[i])+'__x_time.csv')
        x_df = pd.read_csv('TrainingData\subject_00'+str(subjects[i])+'_0'+str(versions[i])+'__x.csv')
        y_df = pd.read_csv('TrainingDataClean\subject_00'+str(subjects[i])+'_0'+str(versions[i])+'__yn.csv')

        xtime_np = xtime_df.values.tolist()
        x_np = x_df.values.tolist()
        y_np = y_df.values.tolist()

        xtime_main.append(xtime_np[:])
        x_main.append(x_np[:])
        y_main.append(y_np[:])

    return xtime_main, x_main, y_main

def BatchData(xtime_main, x_main, y_main, batch_size):

    BATCH_SIZE = batch_size
    temp_xtime = []
    temp_x = []
    temp_y = []
    batches_xtime = []
    batches_x = []
    batches_y = []
    for i in range(len(xtime_main)):
        for j in range(len(xtime_main[i])):
            temp_xtime.append(xtime_main[i][j])
            temp_x.append(x_main[i][j])
            temp_y.append(y_main[i][j][1])
            if (j + 1) % BATCH_SIZE == 0:
                batches_xtime.append(temp_xtime)
                batches_x.append(temp_x)
                batches_y.append(temp_y)
                temp_xtime = []
                temp_x = []
                temp_y = []
        
    return batches_xtime, batches_x, batches_y

'''
Because there are not frequent jumps in the labels across time steps, this function will sample datapoints to decrease training time.
'''
def SampleData(xtime_main, x_main, y_main, freq):

    temp_time = []
    tempx = []
    tempy = []
    sampled_x = []
    sampled_y = []
    sampled_time = []
    for i in range(len(xtime_main)):
        for j in range(len(xtime_main[i])):
            if (j + 1) % freq == 0:
                temp_time.append(xtime_main[i][j])
                tempx.append(x_main[i][j])
                tempy.append(y_main[i][j])
        sampled_time.append(temp_time)
        sampled_x.append(tempx)
        sampled_y.append(tempy)
        temp_time = []
        tempx = []
        tempy = []

    return sampled_x, sampled_y, sampled_time
        
def maxAction(Q, state, actions): #This function spits out the action with the highest value
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

def sample_batches(xbatches, ybatches, num_of_batches): # This function will select batches so that the given data is balanced
    y_batches = []
    x_batches = []

    counter = 0
    for _ in range(num_of_batches):
        sample = random.randint(0, (len(xbatches) - 1)) 
        if 0 in ybatches[sample]:
            if counter < 600:
                y_batches.append(ybatches[sample])
                x_batches.append(xbatches[sample])
                counter+=1
        else:
            y_batches.append(ybatches[sample])
            x_batches.append(xbatches[sample])
        
    
    return x_batches, y_batches

def tokenize(x): # This will tokenize the data for the Q-table
    
    temp2 = []
    vocab_x = []
    vocab_lists = []
    tokens_x = []
    counter = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] not in vocab_x:
                vocab_x.append(counter)
                vocab_lists.append(x[i][j])
                temp2.append(counter)
                counter+=1
        tokens_x.append(temp2)
    
        temp2 = []

    return tokens_x, vocab_x, vocab_lists

if __name__ == '__main__':
    
#TRAINING
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0 # for e-greedy 

    # other hyperparameters
    BATCH_SIZE = 40
    EPISODES = 30
    ITERATIONS = 3000
    FREQ = 10

    # load, process, and batch the data
    time, x, y = TrainingDataLoading() # Load the Data
    sample_x, sample_y, sample_time  = SampleData(time, x, y, FREQ) # sample the data points at set frequency because no observed frequent and/or abrupt jumps
    x_tokens, vocab_x, vocab_lists = tokenize(sample_x) # this will change x lists to ID's in order to make them hashable. 
    time_xbatches, x_batches, y_batches = BatchData(sample_time, x_tokens, y, BATCH_SIZE) # Batches data in size BATCH_SIZE
    sample_xbatches, sample_ybatches = sample_batches(x_batches, y_batches, ITERATIONS)
    
    

    classifier = RL(vocab_x)
    Q = {}
    for state in classifier.stateSpace:
        for action in classifier.actionSpace:
            Q[state, action] = 0 #table of state and action pairs for our Q Learning

    
    totalRewards = np.zeros(ITERATIONS)

    for _ in range(EPISODES):
        for i in range(len(sample_xbatches)):
            action_list = []
            done = False
            epRewards = 0
            observation = sample_xbatches[i][0]
            batch_ind = 0
            while batch_ind < (len(sample_xbatches[i]) - 1): 
                rand = np.random.random()
                action = maxAction(Q,observation, classifier.actionSpace) if rand < (1-EPS) \
                                    else classifier.actionSpaceSample()
                action_list.append(action)
                observation_ = sample_xbatches[i][batch_ind + 1]
                reward = classifier.loss(torch.tensor([sample_ybatches[i][batch_ind + 1]]), torch.tensor([action]))
                epRewards += reward

                action_ = maxAction(Q, observation_, classifier.actionSpace)
                Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                            GAMMA*Q[observation_,action_] - Q[observation,action])
                observation = observation_
                batch_ind+=1

        if EPS - 2 / EPISODES > 0:
            EPS -= 2 / EPISODES
        else:
            EPS = 0
        totalRewards[i] = epRewards

    print("Done")

#TESTING

def TestingDataLoading():
        
    subjects = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8]
    versions = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1]
    xtime_main = []
    x_main = []
    y_main = []

    for i in range(len(subjects)):

        x_df = pd.read_csv('TrainingData\subject_00'+str(subjects[i])+'_0'+str(versions[i])+'__x.csv')

        x_np = x_df.values.tolist()

        
        x_main.append(x_np[:])
        