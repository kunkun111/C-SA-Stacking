# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:05:30 2022

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
from skmultiflow.drift_detection import DDM
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
import time
from matplotlib.font_manager import FontProperties




# Generate the synthetic data samples with incremental change, 5 features and 1000 samples, 1 change per 250 samples.

# Generate sudden data
data = np.zeros((200, 4))
for i in range (200):
    
    m = i % 50
    n = i // 50  # n = 0,1,2,3
    
    x1 = np.random.uniform(0,1)
    x2 = np.random.uniform(0,2)
    x3 = np.random.uniform(0,1)

    # #------------------    
    # # s1 data
    # #------------------
    # if n <= 2:
    #     data[i,3] = 50*0.1 * n + x1+x2
        
    # if n == 3:
    #     data[i,3] = 50*0.1 * (n-2) + x1+x2
        
    # #------------------    
    # # s2 data
    # #------------------
    # if i <= 75:
    #     data[i,3] = 50*0.1 * 0 + x1+x2
    # elif i <= 125:
    #     data[i,3] = 50*0.1 * 1 + x1+x2
    # elif i <= 175:
    #     data[i,3] = 50*0.1 * 2 + x1+x2
    # elif i <= 200:
    #     data[i,3] = 50*0.1 * 1 + x1+x2
       
    # #------------------    
    # # s3 data
    # #------------------
    # if i <= 125:
    #     data[i,3] = 50*0.1 * 0 + x1+x2
    # elif i <= 175:
    #     data[i,3] = 50*0.1 * 1 + x1+x2
    # elif i <= 200:
    #     data[i,3] = 50*0.1 * 2 + x1+x2
        
    # #------------------    
    # # s4 data
    # #------------------
    # if n <= 2:
    #     data[i,3] = 50*0.1 * n*0.5 + x1+x2
        
    # if n == 3:
    #     data[i,3] = 50*0.1 * (n-2)*0.5 + x1+x2
       
    #------------------    
    # s5 data
    #------------------
    if i <= 50:
        data[i,3] = 50*0.1 * 0 + x1+x2
    elif i <= 75:
        data[i,3] = 50*0.1 * 2 + x1+x2
    elif i <= 125:
        data[i,3] = 50*0.1 * 0 + x1+x2
    elif i <= 150:
        data[i,3] = 50*0.1 * 3 + x1+x2
    elif i <= 175:
        data[i,3] = 50*0.1 * 0 + x1+x2
    elif i <= 200:
        data[i,3] = 50*0.1 * 0 + x1+x2
        
    
    # #------------------    
    # # s6 data
    # #------------------
    # if i <= 25:
    #     data[i,3] = 50*0.1 * 0 + x1+x2
    # elif i <= 125:
    #     data[i,3] = 50*0.1 * 1 + x1+x2
    # elif i <= 200:
    #     data[i,3] = 50*0.1 * 2 + x1+x2
        
        
    # #------------------    
    # # s7 data
    # #------------------
    # data[i,3] = 50*0.1 * 0 + x1+x2
       
        
    
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = x3
        
plt.figure(figsize=(8,3))    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(data[:,3], label = '突发型事件风险状态')
plt.axvline(50, color = 'Orange', linestyle = '--')
plt.axvline(100, color = 'Orange', linestyle = '--')
plt.axvline(150, color = 'Orange', linestyle = '--')
plt.xlabel('时刻')
plt.ylabel('事件风险状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper right')
plt.show()

np.savetxt('...//data_sudden5.csv', data, delimiter = ',')



# visualize

data1 = pd.read_csv('D:/SHU/2022桌面/Work6/data_sudden1.csv',  names=['x1', 'x2', 'x3', 'y'])
data2 = pd.read_csv('D:/SHU/2022桌面/work6/data_sudden7.csv',  names=['x1', 'x2', 'x3', 'y'])

data1 = data1.values
data2 = data2.values

plt.rc('font', family='Times New Roman')

plt.figure(figsize=(8,3.2)) 
font1 = FontProperties(fname=r"C:/Users/Administrator/Downloads/simsun/SIMSUN.ttf", size=10)   
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['font.size'] = 10

font2 = {'family':'Times New Roman', 'size':10}

plt.plot(data1[:,3], label = 'S1', c = 'b')
plt.plot(data2[:,3], label = 'S5', c = 'r')

plt.axvline(50, color = 'Lime', linestyle = '--')
plt.axvline(100, color = 'Lime', linestyle = '--')
plt.axvline(150, color = 'Lime', linestyle = '--')
plt.xlabel('时刻', fontproperties = font1)
plt.ylabel('事件风险状态', fontproperties = font1)

plt.ylim(0, 20)
plt.legend(loc = 'upper right', prop=font2)
plt.savefig('D:/SHU/2022桌面/work6/投稿 - Copy/fig/f17.svg', dpi=500)
plt.show()



