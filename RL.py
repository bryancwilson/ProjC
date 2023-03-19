import numpy as np
import pandas as pd
import csv
import random
import torch

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
def BalanceData(xtime_main, x_main, y_main):
    pass

def maxAction(Q, state, actions): #This function spits out the action with the highest value
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

def sample_batches(xbatches, ybatches, num_of_batches):
    y_samples = []
    x_samples = []

    for _ in range(num_of_batches):
        sample = random.randint(0, (len(xbatches) - 1))
        y_samples.append(ybatches[sample])
        x_samples.append(xbatches[sample])

    return x_samples, y_samples

def tokenize(x):
    
    temp2 = []
    vocab_x = []
    tokens_x = []
    counter = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] not in vocab_x:
                vocab_x.append(counter)
                temp2.append(counter)
                counter+=1
        tokens_x.append(temp2)
    
        temp2 = []

    return tokens_x, vocab_x

if __name__ == '__main__':

    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    # other hyperparameters
    BATCH_SIZE = 40
    EPISODES = 30
    ITERATIONS = 3000

    # load and batch the data
    time, x, y = TrainingDataLoading()
    time_xbatches, x_batches, y_batches = BatchData(time, x, y, BATCH_SIZE) 
    sample_xbatches, sample_ybatches = sample_batches(x_batches, y_batches, ITERATIONS)
    x_tokens, vocab_x = tokenize(sample_xbatches)
    

    classifier = RL(vocab_x)
    Q = {}
    for state in classifier.stateSpace:
        for action in classifier.actionSpace:
            Q[state, action] = 0 #table of state and action pairs for our Q Learning

    
    totalRewards = np.zeros(ITERATIONS)

    for _ in range(EPISODES):
        for i in range(ITERATIONS):
            action_list = []
            done = False
            epRewards = 0
            observation = x_tokens[i][0]
            batch_ind = 0
            while batch_ind != (BATCH_SIZE - 1): 
                rand = np.random.random()
                action = maxAction(Q,observation, classifier.actionSpace) if rand < (1-EPS) \
                                    else classifier.actionSpaceSample()
                action_list.append(action)
                observation_ = x_tokens[i][batch_ind + 1]
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