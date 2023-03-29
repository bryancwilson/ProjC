import csv
import pandas as pd
import numpy as np
import torch 

SUBJECT = 1
VERSION = 1
xtime_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__x_time.csv')
x_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__x.csv')
ytime_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__y_time.csv')
y_df = pd.read_csv('TrainingData\subject_00'+str(SUBJECT)+'_0'+str(VERSION)+'__y.csv')
