import csv
import numpy as np
import pandas as pd


def load_data(test_bool = False, groups = 2):
    # THIS LOADS THE TRAINING DATA
    subj_train = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8]
    vers_train = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1]

    x_vectors = []
    for i in range(len(subj_train)):
        x_df_train = pd.read_csv('TrainingData\subject_00'+str(subj_train[i])+'_0'+str(vers_train[i])+'__x.csv')
        x_np_train = x_df_train.values.tolist()
        x_vectors.append(x_np_train[:])
        
    # THIS LOADS THE TESTING DATA
    if test_bool:
        subj_test_tens = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        subj_test_ones = [9, 9, 9, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        vers_test = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for i in range(len(subj_test_tens)):
            x_df_test = pd.read_csv('TestingData\subject_0'+str(subj_test_tens[i])+''+str(subj_test_ones[i])+'_0'+str(vers_test[i])+'__x.csv')
            x_np_test = x_df_test.values.tolist()
            x_vectors.append(x_np_test[:])
    
    # LOAD CLEAN Y's
    subj_y = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8]
    vers_y = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1]

    y_s = []
    for i in range(len(subj_y)):
        y_df = pd.read_csv('TrainingDataClean\subject_00'+str(subj_y[i])+'_0'+str(vers_y[i])+'__yn.csv')
        y_np = y_df.values.tolist()
        y_s.append(y_np[:])

    # COMBINES VECTORS INTO LONGER VECTOR GROUPS
    x_vector_groups = []
    y_sel = []
    for i in range(len(x_vectors)):
        temp  = []
        temp_y = []
        temp_list = []
        for j in range(1, len(x_vectors[i]) - 1):
            temp = temp + x_vectors[i][j]
            if j % groups == 0:
                temp_y.append(y_s[i][j][1])
                temp_list.append(temp)
                temp = []
        x_vector_groups.append(temp_list)
        y_sel.append(temp_y)

    # SAVE PROCESSED DATA AS CSV
    subj = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8]
    vers = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1]
    for i in range(len(subj)):
        my_csv_x = pd.DataFrame(x_vector_groups[i])
        my_csv_y = pd.DataFrame(y_sel[i])
        my_csv_x.to_csv('TrainingDataCleanMLP/subject_00'+str(subj[i])+'_0'+str(vers[i])+'__x.csv')
        my_csv_y.to_csv('TrainingDataCleanMLP/subject_00'+str(subj[i])+'_0'+str(vers[i])+'__yn.csv')

    '''
    # NORMALIZE THE DATA BETWEEN -1 AND 1
    x_list = []
    for i in range(len(x_vectors)):
        for j in range(len(x_vectors[i])):
            for k in range(len(x_vectors[i][j])):
                x_list.append(x_vectors[i][j][k])
    '''

    return x_vector_groups, y_sel


x_vectors, y_sel = load_data(groups = 4)


print('Done')