# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:52:41 2018

@author: weng
"""
import numpy as np
import matplotlib.pyplot as plt


#导入数据
dataMat = []; labelMat = []  
fr = open('C:/Users/weng/Desktop/xg.txt')  
for line in fr.readlines():  
    lineArr = line.strip().split()  
    dataMat.append([float(lineArr[1]), float(lineArr[2])])  
    labelMat.append(int(lineArr[3]))  
    
x=dataMat
y=np.mat(labelMat).T
trainnum1=8
trainnum2=9


mu0=np.mean(x[0:trainnum1], axis=0)
mu1=np.mean(x[trainnum1:], axis=0)
sigma0=np.cov(np.transpose(x[0:trainnum1]))  
sigma0=np.mat(sigma0)
sigma1=np.cov(np.transpose(x[trainnum1:]))  
sigma1=np.mat(sigma1)

sw=sigma0+sigma1
w=np.dot(sw.I,(mu0-mu1))

x=np.mat(x)
plt.plot(x[0:trainnum1,0],x[0:trainnum1,1],'*')
plt.plot(x[trainnum1:,0],x[trainnum1:,1],'+')

x_w=np.linspace(0,0.8,100)
y_w=-w[0,0]*x_w/w[0,1]

plt.plot(x_w,y_w,'--')