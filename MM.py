import csv
import numpy as np
import pandas as pd
import random

# Markoc Chain Model
class MCM:
    def __init__(self):
        self.states = [0, 1, 2, 3]
        self.transMatrix = [[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]
        
    def prob(self):
        pass

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

def BatchData(xtime_main, x_main, y_main):

    BATCH_SIZE = 40
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

def BalanceData(batches_xtime, batches_x, batches_y):
    pass

#==============================================================================================================
# TRAINING BLOCK #

time, x, y = TrainingDataLoading()
time_batches, x_batches, y_batches = BatchData(time, x, y)
NUM_ITERATIONS = 3000 # number of batches you want the model to see when training

markov = MCM()
for iter in range(NUM_ITERATIONS):
    batchChoice = random.randint(0, len(time_batches))
    for i in range(40): # Length of all batches
        rowofTM = markov.transMatrix[y_batches[batchChoice][i]] # the model will pick the row corresponding to the current state
        highestProb = max(rowofTM) # the model will pick the choice with the highest probabilities
        





