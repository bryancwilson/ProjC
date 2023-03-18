import csv
import pandas as pd
import numpy as np


'''
Assumptions:

I am going to assume that the labels do not change between the y labels time steps
'''
SUBJECT = 8
VERSION = 1
xtime_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__x_time.csv')
x_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__x.csv')
ytime_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__y_time.csv')
y_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__y.csv')
print("X Time:", xtime_df)
print("X:", x_df)
print("Y Time:", ytime_df)
print("Y:", y_df)

xtime_np = xtime_df.to_numpy()
ytime_np = ytime_df.to_numpy()
y_np = y_df.to_numpy()

ynew = []
counter = 0
minLength = min(len(xtime_np), len(ytime_np)) # Find the minimum length between x and y times

for i in range(len(xtime_np)):
    if xtime_np[i][0] <= ytime_np[counter][0]:
        ynew.append(y_np[counter][0])
  
    else:
        counter+=1
        if counter < len(y_np):
            ynew.append(y_np[counter][0])
        
        else:
            break

for j in range(len(xtime_np) - i):
    ynew.append(y_np[counter - 1][0])

ynew_df = pd.DataFrame(ynew)
ynew_df.to_csv('TrainingDataClean/subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__yn.csv')
print("Done")
    
