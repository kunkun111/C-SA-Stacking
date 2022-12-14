# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:05:30 2022

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
import seaborn as sns
import time
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor



np.random.seed(0)

# Load data
#---------------------------------

data1 = pd.read_csv('C:/Users/Administrator/Desktop/Work6/data_sudden1.csv',  names=['x1', 'x2', 'x3', 'y'])
data2 = pd.read_csv('C:/Users/Administrator/Desktop/Work6/data_sudden2.csv',  names=['x1', 'x2', 'x3', 'y'])
data1 = data1.values
data2 = data2.values

x_train1 = data1[0:10, :-1]
y_train1 = data1[0:10, -1]

x_train2 = data2[0:10, :-1]
y_train2 = data2[0:10, -1]

x_val1 = data1[10:20, :-1]
y_val1 = data1[10:20, -1]

x_val2 = data2[10:20, :-1]
y_val2 = data2[10:20, -1]

x_train = np.vstack((x_train1, x_train2))
y_train = np.hstack((y_train1, y_train2))

x_val = np.vstack((x_val1, x_val2))
y_val = np.hstack((y_val1, y_val2))


T1 = time.time()


# fit stacking model
#---------------------------------

model1 = GradientBoostingRegressor()
model1.fit(x_train, y_train)

model2 = RandomForestRegressor()
model2.fit(x_train, y_train)

model3 = AdaBoostRegressor()
model3.fit(x_train, y_train)


# validate stacking model
#----------------------------------

y_val_pred1 = model1.predict(x_val)
y_val_pred2 = model2.predict(x_val)
y_val_pred3 = model3.predict(x_val)

meta_x_train = np.vstack((y_val_pred1,y_val_pred2,y_val_pred2))
meta_x_train = meta_x_train.T
meta_y_train = y_val

# print(meta_x_train.shape)
# print(meta_y_train.shape)


# get the relate x, y, and train the related model
#----------------------------------
relate_data = (y_val_pred1 + y_val_pred2 + y_val_pred3)/3

x_train_relate = meta_x_train[0:10, :]
y_train_relate = relate_data[10:20]

# relate_model = GradientBoostingRegressor()
relate_model= RandomForestRegressor()
# relate_model = AdaBoostRegressor()
relate_model.fit(x_train_relate, y_train_relate)


# train the meta model
#----------------------------------

# meta_model = GradientBoostingRegressor()
meta_model = RandomForestRegressor()
# meta_model = AdaBoostRegressor()

meta_model.fit(meta_x_train, meta_y_train)


# get the initial results
#----------------------------------
y_pred_ini = meta_model.predict(meta_x_train)

loss_ini = np.mean(np.square(y_pred_ini - meta_y_train))


# testing and retrain the meta model
#-----------------------------------

stream1 = data1[20:, :]
stream2 = data2[20:, :]

y1 = stream1[:, -1]
y2 = stream2[:, -1]

batch_size = 10
results1 = np.empty(0)
results2 = np.empty(0)
results2_relate = np.empty(0)


for i in range (0, stream1.shape[0], batch_size):
    
    x_test1 = stream1[i:i + batch_size, :-1]
    y_test1 = stream1[i:i + batch_size, -1]
    
    x_test2 = stream2[i:i + batch_size, :-1]
    y_test2 = stream2[i:i + batch_size, -1]
    
    x_test = np.vstack((x_test1, x_test2))
    y_test = np.hstack((y_test1, y_test2))
    
    
    # testing the stacking model
    #-----------------------------------
    
    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)
    y_pred3 = model3.predict(x_test)
    
    y_pred = np.vstack((y_pred1,y_pred2,y_pred2))
    y_pred = y_pred.T
    
    
    # testing and retraining the related model
    #-----------------------------------
    
    y_pred_relate = (y_pred1 + y_pred2 + y_pred3)/3
    
    x_test_relate = y_pred[0:10, :]
    y_test_relate = y_pred_relate[10:20]

    y_pred_final2_relate = relate_model.predict(x_test_relate)
    
    x_train_relate = x_test_relate
    y_train_relate = y_test_relate
    
    relate_model.fit(x_train_relate, y_train_relate)
    
    
    
    # testing and online retraining the meta model
    #----------------------------------- 
    
    y_pred_final = np.zeros((y_pred.shape[0]))
    
    for i in range(y_pred.shape[0]):
        
        y_pred_current = meta_model.predict(y_pred[i, :].reshape(1,-1))
        y_pred_final[i] = y_pred_current
        
        # get current result
        #-----------------------------------
        loss_cur = np.mean(np.square(y_test[i] - y_pred_current))
        
        # retrain meta model
        #-----------------------------------
        if loss_cur >= loss_ini:
            
            meta_x_train = y_pred[i, :].reshape(1,-1)
            meta_y_train = np.array([y_test[i]])
            
            # print(meta_x_train.shape)
            # print(meta_y_train.shape)
            
            meta_model.fit(meta_x_train, meta_y_train)
            
        loss_ini = loss_cur
    
    
    # collect the final predict results
    #-----------------------------------
    
    y_pred_final1 = y_pred_final[0:10]
    y_pred_final2 = y_pred_final[10:20]
    
    results1 = np.hstack((results1, y_pred_final1))
    results2 = np.hstack((results2, y_pred_final2))
    results2_relate = np.hstack((results2_relate, y_pred_final2_relate))
    


# calculate the final results
#--------------------------------

loss_final1 = []
loss_final2 = []
for i in range(y1.shape[0]):
    
    loss1 = np.square(y1[i] - results1[i])
    # loss1 = np.abs(y1[i] - results1[i])
    
    loss2 = np.square(y2[i] - results2[i])
    loss2_relate = np.square(y2[i] - results2_relate[i])
    # loss2 = np.abs(y2[i] - results2[i])
    # loss2_relate = np.abs(y2[i] - results2_relate[i])
    
    
    loss_final1.append(loss1)
    
    if loss2 <= loss2_relate:
        loss_final2.append(loss2)
    else:
        loss_final2.append(loss2_relate)
        

l1_final = np.mean(loss_final1)
l2_final = np.mean(loss_final2)

T2 = time.time()
print('程序运行时间:%s秒' % (T2 - T1))      
    
# print(results)
print('loss1:', l1_final)
print('loss2:', l2_final)

# np.savetxt('MAE_OS4_S2.csv', loss_final2, delimiter = ',')
# np.savetxt('MSE_OS4_S2.csv', loss_final2, delimiter = ',')


# plt.figure(figsize=(6,4))  
# # plt.plot(loss_final1, label = '目标值', c = 'b')
# plt.plot(loss_final2, label = '目标值')
# plt.show()

'''
a = data[:, -1]
b = np.array(final_results)

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3))   
plt.plot(a, label = '目标值', c = 'b')
plt.plot(b, label = '预测值', c = 'r', linestyle = '--')


plt.xlabel('时刻')
plt.ylabel('事件状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()
'''






















